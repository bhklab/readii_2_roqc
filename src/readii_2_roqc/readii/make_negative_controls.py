import itertools
from pathlib import Path

import click
import pandas as pd
import SimpleITK as sitk
from damply import dirs
from imgtools.io.writers.nifti_writer import NIFTIWriter, NiftiWriterIOError
from tqdm import tqdm

from readii.image_processing import alignImages, flattenImage
from readii.io.loaders import loadImageDatasetConfig
from readii.negative_controls_refactor import NegativeControlManager
from readii.process.config import get_full_data_name
from readii.utils import logger
from readii_2_roqc.utils.metadata import get_masked_image_metadata


def get_readii_settings(dataset_config: dict) -> tuple[list, list, list]:
    """Extract READII settings from a configuration dictionary.
    
    Parameters
    ----------
    dataset_config : dict
        Configuration dictionary read in with `loadImageDatasetConfig` containing READII settings
    
    Returns
    -------
    tuple
        A tuple containing:
        - regions: list of regions to process
        - permutations: list of permutations to apply
        - crop: list of crop settings
    """
    readii_config = dataset_config['READII']
    if 'IMAGE_TYPES' not in readii_config:
        message = "READII configuration must contain 'IMAGE_TYPES'."
        logger.error(message)
        raise KeyError(message)
    
    regions = readii_config['IMAGE_TYPES']['regions']

    permutations = readii_config['IMAGE_TYPES']['permutations']

    crop = readii_config['IMAGE_TYPES']['crop']

    return regions, permutations, crop



def save_out_negative_controls(nifti_writer: NIFTIWriter,
                               patient_id: str,
                               image: sitk.Image,
                               region: str,
                               permutation: str,
                               original_image_path: Path,
                               mask_path: Path,
                               mask_image_id: str) -> Path:
    """Save out negative control images using the NIFTIWriter."""
    out_path = None
    try:
        out_path = nifti_writer.save(
                        image,
                        PatientID=patient_id,
                        Region=region,
                        Permutation=permutation,
                        ImageID_mask=mask_image_id.replace(' ', "_"),
                        dir_original_image=original_image_path.parent,
                        dirname_mask=mask_path.parent.name,
                    )
    except NiftiWriterIOError:
        message = f"{permutation} {region} negative control file already exists for {patient_id}. If you wish to overwrite, set overwrite to true in the NIFTIWriter."
        logger.debug(message)

    return out_path



@click.command()
@click.option('--dataset', help='Dataset configuration file name (e.g. NSCLC-Radiomics.yaml). Must be in config/datasets.')
@click.option('--overwrite', help='Whether to overwrite existing readii image files', default=False)
@click.option('--seed', help='Random seed used for negative control generation.', default=10)
def make_negative_controls(dataset: str,
                           overwrite : bool = False,
                           seed: int = 10
                           ) -> list[Path] :
    """Create negative control images and save them out as niftis
    
    Parameters
    ----------
    dataset : str
        Name of the dataset to perform extraction on. Must have a configuration file in the config/datasets directory.
    overwrite : bool = False
        Whether to overwrite existing feature files.
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

    # get path to dataset config directory
    config_dir_path = dirs.CONFIG / 'datasets'
    
    # Load in dataset configuration settings from provided dataset name
    dataset_config = loadImageDatasetConfig(dataset, config_dir_path)

    dataset_name = dataset_config['DATASET_NAME']
    full_data_name = get_full_data_name(config_dir_path / dataset)
    logger.info(f"Creating negative controls for dataset: {dataset_name}")

    # Extract READII settings from config file
    regions, permutations, _crop = get_readii_settings(dataset_config)

    # Set up negative control manager with settings from config
    manager = NegativeControlManager.from_strings(
        negative_control_types=permutations,
        region_types=regions,
        random_seed=seed
    )

    # Get path to the images output by med-imagetools autopipeline run
    mit_images_dir_path = dirs.PROCDATA / full_data_name / 'images' /f'mit_{dataset_name}'
   
    # Load in mit_index file
    dataset_index = pd.read_csv(Path(mit_images_dir_path, f'mit_{dataset_name}_index-simple.csv'))

    # Get just the rows with the desired image and mask modalities and ROIs specified in the dataset config
    image_modality = dataset_config["MIT"]["MODALITIES"]["image"]
    mask_modality = dataset_config["MIT"]["MODALITIES"]["mask"]
    masked_image_index = get_masked_image_metadata(dataset_index = dataset_index,
                                                   dataset_config = dataset_config,
                                                   image_modality = image_modality,
                                                   mask_modality = mask_modality)

    # Set up directory to save out the negative controls
    readii_image_dir = mit_images_dir_path.parent / f'readii_{dataset_name}'

    # Check for index file existence and overwrite status to determine if continuing to negative control creation
    readii_index_file = readii_image_dir / f'readii_{dataset_name}_index.csv'

    if readii_index_file.exists() and not overwrite: 
        # Load in readii index and check:
        # 1. if all negative controls requested have been extracted
        # 2. for all of the patients
        readii_index = pd.read_csv(readii_index_file)

        # Get list of patients that have already been processed and what has been requested based on the dataset index
        processed_samples = set(readii_index['PatientID'].to_list())
        requested_samples = set(dataset_index['PatientID'].to_list())

        processed_image_types = {itype for itype in readii_index[['Permutation', 'Region']].itertuples(index=False, name=None)}
        requested_image_types = {itype for itype in itertools.product([permutation.name() for permutation in manager.negative_control_strategies],
                                                                    [region.name() for region in manager.region_strategies])}
        
        # Check if the requested image types are a subset of those already processed
        if requested_image_types.issubset(processed_image_types) and requested_samples.issubset(processed_samples):
            print("Requested negative controls have already been generated for these samples or are listed in the readii index as if they have been. Set overwrite to true if you want to re-process these.")
            return readii_index['filepath'].to_list()


    if overwrite:
        existing_file_mode = 'OVERWRITE'
        overwrite_index = True
    else:
        existing_file_mode = 'SKIP'
        overwrite_index = False

    # Set up writer for saving out the negative controls and index file
    nifti_writer = NIFTIWriter(
            root_directory = readii_image_dir,
            filename_format = "{dir_original_image}/{dirname_mask}_{ImageID_mask}/" + f"{image_modality}" + "_{Permutation}_{Region}.nii.gz",
            create_dirs = True,
            existing_file_mode = existing_file_mode,
            sanitize_filenames = True,
            index_filename = readii_image_dir /f"readii_{dataset_name}_index.csv",
            overwrite_index = overwrite_index
        )

    # Loop over each study in the masked image index
    for study, study_data in tqdm(masked_image_index.groupby('StudyInstanceUID'), 
                                  desc="Generating READII negative controls", 
                                  total=len(masked_image_index['StudyInstanceUID'].unique())):
        logger.info(f"Processing StudyInstanceUID: {study}")

        # Get image metadata as a pd.Series
        image_metadata = study_data[study_data['Modality'] == image_modality].squeeze()
        try:
            image_path = Path(image_metadata['filepath'])
        except TypeError:
            if image_metadata.empty:
                message = f"No {image_modality} images for study {study}."
                logger.debug(message)
                print(message)
                continue
            else:
                raise
        
        # Load in image
        raw_image = sitk.ReadImage(mit_images_dir_path / image_path)
        # Remove extra dimension of image, set origin, spacing, direction to original
        image = alignImages(raw_image, flattenImage(raw_image))

        # Get mask metadata as a pd.Series
        all_mask_metadata = study_data[study_data['Modality'] == mask_modality]

        # Process each mask for the current study and generate negative control versions of the image
        for _, mask_metadata in tqdm(all_mask_metadata.iterrows(),
                                     desc="Processing each mask for this Study",
                                     total=len(all_mask_metadata)):
            # Get path to the mask image file
            try:
                mask_path = Path(mask_metadata['filepath'])
            except TypeError:
                if mask_metadata.empty:
                    message = f"No {mask_modality} masks for study {study}."
                    logger.debug(message)
                    print(message)
                    continue
                else:
                    raise
            # Load in mask
            raw_mask = sitk.ReadImage(mit_images_dir_path / mask_path)
            mask = alignImages(raw_mask, flattenImage(raw_mask))

            # Generate each image type and save it out with the nifti writer
            readii_image_paths = [save_out_negative_controls(nifti_writer, 
                                                             patient_id = image_metadata['PatientID'],
                                                             image = neg_image,
                                                             region = region,
                                                             permutation = permutation,
                                                             original_image_path = image_path,
                                                             mask_image_id = mask_metadata['ImageID'],
                                                             mask_path = mask_path
                                        ) for neg_image, permutation, region in manager.apply(image, mask)]

    return readii_image_paths



if __name__ == '__main__':
    make_negative_controls()