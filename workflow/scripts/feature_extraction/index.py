import pandas as pd
import click
from damply import dmpdirs
from pathlib import Path

from readii.io.loaders import loadImageDatasetConfig
from readii.utils import logger
from readii.process.config import get_full_data_name


def generate_pyradiomics_index(image_directory:Path,
                               output_file_path:Path
                               ) -> pd.DataFrame:
    """Set up index file for PyRadiomics feature extraction. Output file contains columns for ID, Image, and Mask.
        - ID = patient identifier
        - Image = path to image file to load for feature extraction, such as a CT or MRI
        - Mask = path to binary label mask file to load for feature extraction, such as a segmentation
    
    Parameters
    ----------
    image_directory : Path

    output_file_path : Path

    Returns
    -------
    dataset_index : pd.DataFrame
    """
    # Construct file path lists for images and masks
    image_files = sorted(image_directory.rglob(pattern="*/CT*/CT.nii.gz"))
    mask_files = sorted(image_directory.rglob(pattern="*/RT*/GTV.nii.gz"))

    # Get list of sample IDs from top of data directory
    unique_sample_ids = [sample_dir.name for sample_dir in sorted(image_directory.glob(pattern="*/"))]

    if len(mask_files) > len(image_files):
        mask_index = pd.DataFrame(data= {'ID': [mask_path.parent.parent.stem for mask_path in mask_files],
                                         'Mask': mask_files})
        image_index = pd.DataFrame(data = {'ID': unique_sample_ids, 'Image': image_files})
        dataset_index = image_index.merge(mask_index, how='outer', left_on='ID', right_on='ID')

    else:
        # Construct dataframe to iterate over
        dataset_index = pd.DataFrame(data = {'ID': unique_sample_ids, 'Image': image_files, 'Mask': mask_files})

    dataset_index.to_csv(output_file_path, index=False)

    return dataset_index



@click.command()
@click.option('--config', help='Dataset configuration file name (e.g. NSCLC-Radiomics.yaml). Must be in config/datasets.')
@click.option('--method', default='pyradiomics', help='Feature extraction method to use.')
def generate_dataset_index(config, method):
    """
    """
    # Load in dataset configuration settings from provided file
    dataset_config = loadImageDatasetConfig(config) 

    data_name = get_full_data_name(dataset_config)
    
    # Construct image directory from DMP and config
    image_directory = dmpdirs.PROCDATA / data_name / "images" / f"mit+{data_name}"

    # Construct output file path from DMP and feature extraction type
    feature_extraction_type = method
    output_file_path = dmpdirs.PROCDATA / data_name / "features" / feature_extraction_type / f"{feature_extraction_type}_{data_name}_index.csv"

    match feature_extraction_type:
        case "pyradiomics":
            dataset_index = generate_pyradiomics_index(image_directory, output_file_path)
        case _:
            message = f"Index generator doesn't exist for {feature_extraction_type}."
            logger.debug(message)
            raise ValueError(message)
    
    return dataset_index



if __name__ == "__main__":
    generate_dataset_index()
    


