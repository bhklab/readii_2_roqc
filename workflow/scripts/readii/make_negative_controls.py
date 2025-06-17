import click
import SimpleITK as sitk
import pandas as pd

from pathlib import Path
from damply import dirs
from joblib import Parallel, delayed
from typing import Optional

from imgtools.io.writers.nifti_writer import NIFTIWriter, NiftiWriterIOError
from readii.image_processing import flattenImage, alignImages
from readii.io.loaders import loadImageDatasetConfig
from readii.negative_controls_refactor import NegativeControlManager
from readii.process.config import get_full_data_name
from readii.utils import logger



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


def get_masked_image_metadata(dataset_index:pd.DataFrame,
                              dataset_config:Optional[dict] = None,
                              image_modality:Optional[str] = None,
                              mask_modality:Optional[str] = None):
    """Get rows of Med-ImageTools index.csv with the mask modality and the corresponding image modality and create a new index with just these rows for READII
    
    Parameters
    ----------
    dataset_index : pd.DataFrame
        DataFrame loaded from a Med-ImageTools index.csv containing image metadata. Must have columns for Modality, ReferencedSeriesUID, and SeriesInstanceUID.
    dataset_config : Optional[dict]
        Dictionary of configuration settings to get image and mask modality from for filtering dataset_index. Must include MIT MODALITIES image and MIT MODALITIES mask. Expected output from running loadImageDatasetConfig.
    image_modality : Optional[str]
        Image modality to filter dataset_index with. Will override dataset_config setting.
    mask_modality : Optional[str]
        Mask modality to filter dataset_index with. Will override dataset_config setting.

    Returns
    -------
    pd.DataFrame
        Subset of the dataset_index with just the masks and their reference images' metadata.
    """

    if image_modality is None:
        if dataset_config is None:
            message = "No image modality setting passed. Must pass a image_modality or dataset_config with an image modality setting."
            logger.error(message)
            raise ValueError(message)
        
        # Get the image modality from config to retrieve from the metadata
        image_modality = dataset_config["MIT"]["MODALITIES"]["image"]
    
    if mask_modality is None:
        if dataset_config is None:
            message = "No mask modality setting passed. Must pass a mask_modality or dataset_config with a mask modality setting."
            logger.error(message)
            raise ValueError(message)
        
        # Get the mask modality from config to retrieve from the metadata
        mask_modality = dataset_config["MIT"]["MODALITIES"]["mask"]

    # Get all metadata rows with the mask modality
    mask_metadata = dataset_index[dataset_index['Modality'] == mask_modality]

    # Get a Series of ReferenceSeriesUIDs from the masks - these point to the images the masks were made on
    referenced_series_ids = mask_metadata['ReferencedSeriesUID']
    
    # Get image metadata rows with a SeriesInstanceUID matching one of the ReferenceSeriesUIDS of the masks
    image_metadata = dataset_index[dataset_index['Modality'] == image_modality]
    masked_image_metadata = image_metadata[image_metadata['SeriesInstanceUID'].isin(referenced_series_ids)]

    # Return the subsetted metadata
    return pd.concat([masked_image_metadata, mask_metadata], sort=True)



def save_out_negative_controls(nifti_writer: NIFTIWriter,
                               patient_id: str,
                               image: sitk.Image,
                               region: str,
                               permutation: str,
                               original_image_path: Path,
                               mask_path: Path,
                               mask_image_id: str) -> Path:
    """Save out negative control images using the NIFTIWriter."""

    try:
        out_path = nifti_writer.save(
                        image,
                        PatientID=patient_id,
                        Region=region,
                        Permutation=permutation,
                        ImageID_mask=mask_image_id,
                        dir_original_image=original_image_path.parent,
                        dirname_mask=mask_path.parent.name,
                    )
    except NiftiWriterIOError as e:
        message = f"{permutation} {region} negative control file already exists for {patient_id}. If you wish to overwrite, set overwrite to true in the NIFTIWriter."
        logger.debug(message)

    return out_path



@click.command()
@click.option('--dataset', help='Dataset configuration file name (e.g. NSCLC-Radiomics.yaml). Must be in config/datasets.')
@click.option('--overwrite', help='Whether to overwrite existing readii image files', default=False)
@click.option('--seed', help='Random seed used for negative control generation.', default=10)
def make_negative_controls(dataset: str,
                           overwrite : bool = False,
                           seed: int = 10):
    """Create negative control images and save them out as niftis"""

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
    dataset_index = pd.read_csv(Path(mit_images_dir_path, f'mit_{dataset_name}_index.csv'))

    # Get just the rows with the desired image and mask modalities specified in the dataset config
    image_modality = dataset_config["MIT"]["MODALITIES"]["image"]
    mask_modality = dataset_config["MIT"]["MODALITIES"]["mask"]
    masked_image_index = get_masked_image_metadata(dataset_index = dataset_index,
                                                   image_modality = image_modality,
                                                   mask_modality = mask_modality)

    # Set up directory to save out the negative controls
    readii_image_dir = mit_images_dir_path.parent / f'readii_{dataset_name}'

    # Check for index file existence and overwrite status to determine if continuing to negative control creation
    readii_index_file = readii_image_dir / f'readii_{dataset_name}_index.csv'
    if readii_index_file.exits() and not overwrite:
        logger.info("READII index file present and no overwrite requested. Skipping negative control generation.")
        return
    
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
    for study, study_data in masked_image_index.groupby('StudyInstanceUID'):
        logger.info(f"Processing StudyInstanceUID: {study}")

        # Get image metadata as a pd.Series
        image_metadata = study_data[study_data['Modality'] == image_modality].squeeze()
        image_path = Path(image_metadata['filepath'])
        # Load in image
        raw_image = sitk.ReadImage(mit_images_dir_path / image_path)
        # Remove extra dimension of image, set origin, spacing, direction to original
        image = alignImages(raw_image, flattenImage(raw_image))

        # Get mask metadata as a pd.Series
        all_mask_metadata = study_data[study_data['Modality'] == mask_modality]

        # Process each mask for the current study and generate negative control versions of the image
        for row_idx, mask_metadata in all_mask_metadata.iterrows():
            # Get path to the mask image file
            mask_path = Path(mask_metadata['filepath'])
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

    return



if __name__ == '__main__':
    make_negative_controls()