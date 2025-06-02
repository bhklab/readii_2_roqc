import click
import SimpleITK as sitk
import pandas as pd

from pathlib import Path
from damply import dirs
from joblib import Parallel, delayed

from readii.image_processing import flattenImage, alignImages
from readii.io.loaders import loadImageDatasetConfig
from readii.io.writers.nifti_writer import NIFTIWriter, NiftiWriterIOError
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


def save_out_negative_controls(nifti_writer: NIFTIWriter,
                               patient_id: str,
                               image: sitk.Image,
                               region: str,
                               permutation: str):
    """Save out negative control images using the NIFTIWriter."""

    try:
        nifti_writer.save(
                        image,
                        PatientID=patient_id,
                        region=region,
                        permutation=permutation
                    )
    except NiftiWriterIOError as e:
        message = f"{permutation} {region} negative control file already exists for {patient_id}. If you wish to overwrite, set overwrite to true in the NIFTIWriter."
        logger.debug(message)

    return image


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

    config_dir_path = dirs.CONFIG / 'datasets'
    
    dataset_config = loadImageDatasetConfig(dataset, config_dir_path)

    dataset_name = dataset_config['DATASET_NAME']
    full_data_name = get_full_data_name(config_dir_path / dataset)
    logger.info(f"Creating negative controls for dataset: {dataset_name}")

    # Extract READII settings
    regions, permutations, _crop = get_readii_settings(dataset_config)

    # Set up negative control manager with settings from config
    manager = NegativeControlManager.from_strings(
        negative_control_types=permutations,
        region_types=regions,
        random_seed=seed
    )

    mit_images_dir_path = dirs.PROCDATA / full_data_name / 'images' /f'mit_{dataset_name}'
   
    dataset_index = pd.read_csv(Path(mit_images_dir_path, f'mit_{dataset_name}_index.csv'))

    image_modality = dataset_config["MIT"]["MODALITIES"]["image"]
    mask_modality = dataset_config["MIT"]["MODALITIES"]["mask"]

    # StudyInstanceUID
    for study, study_data in dataset_index.groupby('StudyInstanceUID'):
        logger.info(f"Processing StudyInstanceUID: {study}")

        # Get image metadata as a pd.Series
        image_metadata = study_data[study_data['Modality'] == image_modality].squeeze()
        image_path = Path(image_metadata['filepath'])
        # Load in image
        raw_image = sitk.ReadImage(mit_images_dir_path / image_path)
        # Remove extra dimension of image, set origin, spacing, direction to original
        image = alignImages(raw_image, flattenImage(raw_image))

        # Get mask metadata as a pd.Series
        mask_metadata = study_data[study_data['Modality'] == mask_modality].squeeze()
        mask_path = Path(mask_metadata['filepath'])
        # Load in mask
        raw_mask = sitk.ReadImage(mit_images_dir_path / mask_path)
        mask = alignImages(raw_mask, flattenImage(raw_mask))

        # Set up writer for saving out the negative controls
        nifti_writer = NIFTIWriter(
            root_directory = mit_images_dir_path.parent / f'readii_{dataset_name}' / image_path.parent,
            filename_format = "{permutation}_{region}.nii.gz",
            overwrite = overwrite,
            create_dirs = True
        )
        
        # Generate each image type and save it out with the nifti writer
        Parallel(n_jobs=-1, require="sharedmem")(
            delayed(save_out_negative_controls)(
                nifti_writer, 
                patient_id = image_metadata['PatientID'],
                image = neg_image,
                region = region,
                permutation = permutation
            ) for neg_image, permutation, region in manager.apply(image, mask)
        )
                
    return



if __name__ == '__main__':
    make_negative_controls()