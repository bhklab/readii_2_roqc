import click
import pandas as pd
import itertools
import SimpleITK as sitk

from damply import dirs
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed

from imgtools.io.writers.nifti_writer import NIFTIWriter, NiftiWriterIOError

from readii.process.images.crop import crop_and_resize_image_and_mask
from readii.negative_controls_refactor.manager import NegativeControlManager
from readii.utils import logger

from readii_2_roqc.utils.loaders import load_dataset_config, load_image_and_mask
from readii_2_roqc.utils.metadata import get_masked_image_metadata, insert_SampleID, make_edges_df
from readii_2_roqc.utils.settings import get_readii_settings, get_resize_string, get_readii_index_filepath

def negative_control_generator(sample_id:str,
                               image:sitk.Image,
                               mask:sitk.Image,
                               regions, 
                               permutations, 
                               crop, 
                               resize,
                               image_meta_id,
                               mask_meta_id,
                               nifti_writer:NIFTIWriter,
                               seed = 10
                               ):
    negative_control_image_paths = []

    # Set up negative control manager with settings from config
    manager = NegativeControlManager.from_strings(
        negative_control_types=permutations,
        region_types=regions,
        random_seed=seed
    )
    # Process and save negative control images
    for proc_image, permutation, region in manager.apply(image, mask):
        # apply crop and resize
        if crop != "" and resize != []:
            proc_image, proc_mask = crop_and_resize_image_and_mask(proc_image, 
                                                                   mask, 
                                                                   crop_method = crop, 
                                                                   resize_dimension = resize)
        # save out negative controls
        try:
            # The capitalized arguments are here on purpose to manipulate the order of the columns in the index file
            out_path = nifti_writer.save(
                            proc_image,
                            ImageID = image_meta_id,
                            MaskID = mask_meta_id,
                            Permutation=permutation,
                            Region=region,
                            Resize=get_resize_string(resize),
                            SampleID=sample_id,
                            crop=crop
                        )
        except NiftiWriterIOError:
            message = f"{permutation} {region} negative control file already exists for {sample_id}. If you wish to overwrite, set overwrite to True."
            logger.debug(message)
        
        negative_control_image_paths.append(out_path)

    return negative_control_image_paths




def image_preprocessor(dataset_config:dict, 
                        image_path:Path, 
                        mask_path:Path, 
                        images_dir_path:Path, 
                        output_dir:Path, 
                        sample_id:str = None, 
                        mask_image_id:str = None, 
                        overwrite:bool = False,
                        seed = 10):
    dataset_name = dataset_config['DATASET_NAME']

    # Get READII image preprocessing settings from config file
    regions, permutations, crop, resize = get_readii_settings(dataset_config)
    # Make a string version of the resize argument for file writing
    resize_string = get_resize_string(resize)

    # Get sample metadata from path if not provided
    if sample_id is None:
        sample_id = Path(image_path).parts[0]
    if mask_image_id is None:
        mask_image_id = Path(mask_path).name.removesuffix('.nii.gz')
    # make image identifier string for file writer
    image_meta_id = f"{image_path.parent.name}"
    # make mask identifier string for file writer
    mask_meta_id = f"{mask_path.parent.name}/{mask_image_id.replace(' ', "_")}"
    
    # Get beginning of the path to the nifti images dir
    mit_images_dir = images_dir_path / f'mit_{dataset_name}'
    # load in the nifti image and mask files, flattened to 3D and aligned with each other
    image, mask = load_image_and_mask(mit_images_dir / image_path, mit_images_dir / mask_path)
    # get image modality for file writer
    image_modality = dataset_config['MIT']['MODALITIES']['image']
    

    # Set up the readii subdirectory for the image being processed, specifically the crop and resize level
    if crop == '' and resize == []:
        # get the original image size to use for output directory, without the slice count
        image_size_string = get_resize_string(image.GetSize()[0:2])
        # make the slice index an n since different images have different slice counts
        crop_setting_string = f'original_{image_size_string}_n'
    else:
        crop_setting_string = f'{crop}_{resize_string}'

    # Set up writer for saving out the negative controls and index file
    if overwrite:
        existing_file_mode = 'OVERWRITE'
    else:
        existing_file_mode = 'SKIP'

    nifti_writer = NIFTIWriter(
            root_directory = output_dir,
            filename_format = f"{crop_setting_string}/{image_path.parent}/{mask_meta_id}/{image_modality}" + "_{Permutation}_{Region}.nii.gz",
            create_dirs = True,
            existing_file_mode = existing_file_mode,
            sanitize_filenames = True,
            index_filename = output_dir / crop_setting_string / f"readii_{dataset_name}_index.csv",
            overwrite_index = False
        )
    
    readii_image_paths = []
    # Process crop and resize of original image if needed, and save
    if crop != '' and resize != []:
        logger.info("Making cropped version of original image")
        crop_image, crop_mask = crop_and_resize_image_and_mask(image, 
                                                               mask, 
                                                               crop_method = crop, 
                                                               resize_dimension = resize)
        # save out cropped image
        try:
            # The capitalized arguments are here on purpose to manipulate the order of the columns in the index file
            out_path = nifti_writer.save(
                            crop_image,
                            ImageID = image_meta_id,
                            MaskID = mask_meta_id,
                            Permutation="original",
                            Region="full",
                            Resize=resize_string,
                            SampleID=sample_id,
                            crop=crop
                        )
        except NiftiWriterIOError:
            message = f"{crop} {resize_string} original image file already exists for {sample_id}. If you wish to overwrite, set overwrite to True."
            logger.debug(message)
        
        readii_image_paths.append(out_path)
    # end original image processing

    if permutations != [] and regions != []:
        logger.info("Making negative control images")
        negative_control_image_paths = negative_control_generator(sample_id=sample_id,
                                                                  image = image,
                                                                  mask = mask,
                                                                  regions = regions,
                                                                  permutations = permutations,
                                                                  crop = crop,
                                                                  resize = resize,
                                                                  image_meta_id = image_meta_id,
                                                                  mask_meta_id = mask_meta_id,
                                                                  nifti_writer = nifti_writer,
                                                                  seed = seed)
        readii_image_paths.append(negative_control_image_paths)
    
    return readii_image_paths




@click.command()
@click.argument('dataset')
@click.option('--overwrite', help='Whether to overwrite existing readii image files', default=False)
@click.option('--parallel', type=click.BOOL, help='Whether to run READII preprocessing in parallel', default=False)
@click.option('--jobs', type=click.INT, help="Number of jobs to give parallel processor", default=-1)
@click.option('--seed', help='Random seed used for negative control generation.', default=10)
def make_negative_controls(dataset: str,
                           overwrite: bool = False,
                           parallel: bool = False,
                           jobs: int = -1,
                           seed: int = 10
                           ) -> list[Path] :
    """Create negative control images for dataset and save them out as niftis
    
    Parameters
    ----------
    dataset : str
        Name of the dataset to perform extraction on. Must have a configuration file in the config/datasets directory.
    overwrite : bool = False
        Whether to overwrite existing feature files.
    parallel : bool = False
        Whether to run READII preprocessing in parallel.
    jobs : int = -1
        Number of jobs to give parallel processor.
    seed : int = 10
        Random seed to use for negative control generation.
    
    Returns
    -------
    readii_image_paths : list[Path]
        List of paths to the saved out negative control NIfTI files.
    """
    if dataset is None:
        message = "Dataset name must be provided."
        logger.error(message)
        raise ValueError(message)

    # Load dataset config file
    dataset_config, dataset_name, full_dataset_name = load_dataset_config(dataset)
    logger.info(f"Creating negative controls for dataset: {dataset_name}") 
    
    # Set up the path to the images data directory
    images_dir_path = dirs.PROCDATA / full_dataset_name / 'images'

    # Load the med-imagetools simple index file for list of files to process
    dataset_index = pd.read_csv(images_dir_path / f'mit_{dataset_name}' / f'mit_{dataset_name}_index-simple.csv')
    # Create a SampleID column by combining the patient ID and sample number - this will be the main identifier
    dataset_index = insert_SampleID(dataset_index)

    # Filter the data index based on the image modalities provided in the config settings
    masked_image_index = get_masked_image_metadata(dataset_index, dataset_config)

    # Load the requested image processing settings from configuration
    regions, permutations, crop, resize = get_readii_settings(dataset_config)

    # Set up the base output directory for the processed images
    readii_image_dir = images_dir_path / f'readii_{dataset_name}'
    

    # Check for existing outputs from this function
    readii_index_filepath = get_readii_index_filepath(dataset_config, readii_image_dir)
    if readii_index_filepath.exists() and not overwrite:
        # Load in readii index and check:
        # 1. if all negative controls requested have been extracted
        # 2. for all of the patients
        readii_index = pd.read_csv(readii_index_filepath)

        # Get list of patients that have already been processed and what has been requested based on the dataset index
        processed_samples = set(readii_index['SampleID'].to_list())
        requested_samples = set(dataset_index['SampleID'].to_list())

        # Check if all the settings columns are present - handles old READII outputs
        readii_settings = ['Permutation', 'Region', 'crop', 'Resize']
        if not set(readii_index.columns).issuperset(readii_settings):
            print("Not all READII settings satisfied in existing output. Re-running negative control generation.")
            overwrite = True
        
        else:
            # Get all combinations of negative control settings that have already been processed and what has been requested in the config file
            processed_image_types = {itype for itype in readii_index[readii_settings].itertuples(index=False, name=None)}
            requested_image_types = {itype for itype in itertools.product(permutations,
                                                                        regions,
                                                                        [crop],
                                                                        [get_resize_string(resize)])}
            
            #TODO: add a function to remove processed samples from the list to process again

            # If everything matches, no processing required
            if requested_image_types.issubset(processed_image_types) and requested_samples.issubset(processed_samples):
                print("All requested negative controls have already been generated for these samples or are listed in the readii index as if they have been. Set overwrite to true if you want to re-process these.")
                return readii_index['filepath'].to_list()
    # end existence checking
    elif readii_index_filepath.exists() and overwrite:
        # If overwriting existing files, delete the existing index file
        # Doing this instead of using overwrite index in the niftiwriter because then it makes a new index file for each sample
        readii_index_filepath.unlink()

    # Make a single row for every image and mask pair to iterate over
    edges_index = make_edges_df(masked_image_index, dataset_config['MIT']['MODALITIES']['image'], dataset_config['MIT']['MODALITIES']['mask'])

    if parallel:
        # Use joblib to parallelize negative control generation
        readii_image_paths = Parallel(n_jobs=jobs)(
                                delayed(image_preprocessor)(
                                    dataset_config=dataset_config, 
                                    image_path=Path(data_row.filepath_image), 
                                    mask_path=Path(data_row.filepath_mask), 
                                    images_dir_path=images_dir_path, 
                                    output_dir=readii_image_dir,
                                    sample_id=data_row.SampleID_image,
                                    mask_image_id=data_row.ImageID_mask, 
                                    overwrite=overwrite
                                )
                                for _, data_row in tqdm(
                                    edges_index.iterrows(),
                                    desc="Generating negative controls for each image-mask pair...",
                                    total=len(edges_index)
                                )
                            )
    else:
        readii_image_paths = [image_preprocessor(dataset_config=dataset_config, 
                                                image_path=Path(data_row.filepath_image), 
                                                mask_path=Path(data_row.filepath_mask), 
                                                images_dir_path=images_dir_path, 
                                                output_dir=readii_image_dir,
                                                sample_id=data_row.SampleID_image,
                                                mask_image_id=data_row.ImageID_mask, 
                                                overwrite=overwrite
                                                ) for _, data_row in tqdm(edges_index.iterrows(),
                                                                            desc="Generating negative controls for each image-mask pair...",
                                                                            total=len(edges_index))]

    return readii_image_paths



if __name__ == '__main__':
    make_negative_controls()