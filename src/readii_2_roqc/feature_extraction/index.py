from pathlib import Path

import click
import pandas as pd
from damply import dirs
from readii.io.loaders import loadImageDatasetConfig
from readii.process.config import get_full_data_name
from readii.utils import logger

from readii_2_roqc.utils.metadata import get_masked_image_metadata, make_edges_df
from readii_2_roqc.utils.settings import get_readii_settings, get_resize_string, get_readii_index_filepath

def get_base_index(dataset_config: dict,
                   mit_index: pd.DataFrame):
    """Set up default index dataframe for feature extraction.
    
    Parameters
    ----------
    dataset_config : dict
        Configuration settings for a dataset, loaded with loadImageDatasetConfig
    mit_index : pd.DataFrame
        Dataframe containing metadata for the images and masks processed by imgtools autopipeline.
    
    Returns
    -------
    base_index : pd.DataFrame
        Dataframe with columns:
        * SampleID: PatientID + SampleNumber from imgtools autopipeline
        * Image - path to the image nifti file
        * Mask - path to the mask nifti file
        * DatasetName - name of the dataset these samples come from
        * SeriesInstanceUID_Image - Series ID from DICOM header for image
        * Modality_Image - image modality (e.g. CT, MR)
        * SeriesInstanceUID_Mask - Series ID from DICOM header for mask
        * Modality_Mask - mask modality (e.g. SEG, RTSTRUCT)
        * MaskID - name of region of interest (ROI) in the mask (e.g. GTVp)
        * readii_Permutation - permutation used for READII negative control
        * readii_Region - region used for READII negative control
    """

    dataset_name = dataset_config['DATASET_NAME']

    image_modality = dataset_config["MIT"]["MODALITIES"]["image"]
    mask_modality = dataset_config["MIT"]["MODALITIES"]["mask"]

    mit_edges_index = make_edges_df(mit_index, image_modality, mask_modality)

    # Set up the data from the mit index to point to the original images for feature extraction
    return pd.DataFrame(data={"SampleID": mit_edges_index.apply(lambda x: f"{x.PatientID}_{str(x.SampleNumber).zfill(4)}", axis=1),
                              "Image": mit_edges_index.apply(lambda x: f"{Path(f'mit_{dataset_name}') / x.filepath_image}", axis=1),
                              "Mask": mit_edges_index.apply(lambda x: f"{Path(f'mit_{dataset_name}') / x.filepath_mask}", axis=1),
                              "DatasetName": dataset_name,
                              "SeriesInstanceUID_Image": mit_edges_index['SeriesInstanceUID_image'],
                              "Modality_Image": mit_edges_index['Modality_image'],
                              "SeriesInstanceUID_Mask": mit_edges_index['SeriesInstanceUID_mask'],
                              "Modality_Mask": mit_edges_index['Modality_mask'],
                              "MaskID": mit_edges_index['ImageID_mask'].replace(' ', '_'),
                              "readii_Permutation": "original",
                              "readii_Region": "full"
                             }
                       )



def get_readii_index(dataset_config: dict,
                     readii_index: pd.DataFrame):
    """Set up index dataframe for feature extraction on READII negative control images using the index file generated from negative control generation.
    
    Parameters
    ----------
    dataset_config : dict
        Configuration settings for a dataset, loaded with loadImageDatasetConfig
    readii_index : pd.DataFrame
        Dataframe containing metadata for the images and masks processed by READII negative control generation.
    
    Returns
    -------
    base_index : pd.DataFrame
        Dataframe with columns:
        * SampleID: PatientID + SampleNumber from imgtools autopipeline
        * Image - path to the image nifti file
        * Mask - path to the mask nifti file
        * DatasetName - name of the dataset these samples come from
        * SeriesInstanceUID_Image - Series ID from DICOM header for image
        * Modality_Image - image modality (e.g. CT, MR)
        * SeriesInstanceUID_Mask - Series ID from DICOM header for mask
        * Modality_Mask - mask modality (e.g. SEG, RTSTRUCT)
        * MaskID - name of region of interest (ROI) in the mask (e.g. GTVp)
        * readii_Permutation - permutation used for READII negative control
        * readii_Region - region used for READII negative control
    """
    dataset_name = dataset_config['DATASET_NAME']

    image_modality = dataset_config["MIT"]["MODALITIES"]["image"]
    mask_modality = dataset_config["MIT"]["MODALITIES"]["mask"]
    

    return pd.DataFrame(data={"SampleID": readii_index.SampleID,
                              "Image": readii_index.apply(lambda x: f"{Path(f'readii_{dataset_name}') / x.filepath}", axis=1),
                              "Mask": readii_index.apply(lambda x: f"{Path(f'mit_{dataset_name}') / x.SampleID / f'{x.MaskID}.nii.gz'}", axis=1),
                              "DatasetName": dataset_name,
                              "SeriesInstanceUID_Image": "",
                              "Modality_Image": image_modality,
                              "SeriesInstanceUID_Mask": "",
                              "Modality_Mask": mask_modality,
                              "MaskID": readii_index.apply(lambda x: f"{Path(x.MaskID).name}", axis=1),
                              "readii_Permutation": readii_index["Permutation"],
                              "readii_Region": readii_index["Region"],
                              "readii_Crop": readii_index["crop"],
                              "readii_Resize": readii_index["Resize"]
                             }
                       )



def generate_pyradiomics_index(dataset_config: dict,
                               mit_index: pd.DataFrame,
                               readii_index: pd.DataFrame | None = None,
                               output_file_path: Path | None = None
                               ) -> pd.DataFrame:
    """Set up and save out index file for PyRadiomics feature extraction. Output file contains columns for ID, Image, and Mask.
    
    Parameters
    ----------
    dataset_config : dict
        Configuration settings for a dataset, loaded with loadImageDatasetConfig
    mit_index : pd.DataFrame
        Dataframe containing metadata for the images and masks processed by imgtools autopipeline.
    readii_index : pd.DataFrame | None
        Dataframe containing metadata for the negative control images processed by make_negative_controls.py using READII
        If not supplied, will set up the index for the original images only.
    output_file_path : Path | None
        File path to save the PyRadiomics index csv out to. If not provided, will be set up as 
        `dirs.PROCDATA / f"{dataset_config['DATA_SOURCE']}_{dataset_name}" / "features" / f"pyradiomics_{dataset_name}_index.csv`

    Returns
    -------
    pyradiomics_index : pd.DataFrame
        Dataframe with columns:
        * SampleID: PatientID + SampleNumber from imgtools autopipeline
        * Image - path to the image nifti file
        * Mask - path to the mask nifti file
        * DatasetName - name of the dataset these samples come from
        * SeriesInstanceUID_Image - Series ID from DICOM header for image
        * Modality_Image - image modality (e.g. CT, MR)
        * SeriesInstanceUID_Mask - Series ID from DICOM header for mask
        * Modality_Mask - mask modality (e.g. SEG, RTSTRUCT)
        * MaskID - name of region of interest (ROI) in the mask (e.g. GTVp)
        * readii_Permutation - permutation used for READII negative control
        * readii_Region - region used for READII negative control
    """
    dataset_name = dataset_config['DATASET_NAME']

    original_images_index = get_base_index(dataset_config, mit_index)

    if readii_index is not None:
        # Set up the data from the readii index to point to the negative control images for feature extraction
        readii_images_index = get_readii_index(dataset_config, readii_index)

        # Concatenate the original and negative control image index dataframes
        pyradiomics_index = pd.concat([original_images_index, readii_images_index], ignore_index=True, axis=0)

        # Sort the resulting index by negative control settings, then SampleID and MaskID
        pyradiomics_index = pyradiomics_index.sort_values(by=['readii_Permutation', 'readii_Region', 'SampleID', 'MaskID'], ignore_index=True)

    else:
        # No negative control images to process, just use original images index
        pyradiomics_index = original_images_index

    try:
        # If no output file path is provided, use default path setup, which makes a features directory for the dataset and saves the index there
        if output_file_path is None:
            output_file_path = dirs.PROCDATA / f"{dataset_config['DATA_SOURCE']}_{dataset_name}" / "features" / f"pyradiomics_{dataset_name}_index.csv"
        
        # Check that output file path is a .csv
        assert output_file_path.suffix == ".csv"

        # Create any missing parent directories for the output
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Save out the index file
        pyradiomics_index.to_csv(output_file_path, index=False)
    
    except AssertionError:
        message = f"output_file_path for generate_pyradiomics_index does not end in .csv. Path given: {output_file_path}"
        logger.error(message)
        raise

    return pyradiomics_index



def generate_fmcib_index(dataset_config: dict,
                         mit_index: pd.DataFrame,
                         readii_index: pd.DataFrame | None = None,
                         output_file_path: Path | None = None
                        ) -> pd.DataFrame:
    """Set up and save out index file for Foundation Model for Cancer Image Biomarkers (FMCIB) feature extraction. Output file contains columns for "image_path", "coordX", "coordY", "coordZ".
    
    Parameters
    ----------
    dataset_config : dict
        Configuration settings for a dataset, loaded with loadImageDatasetConfig
    mit_index : pd.DataFrame
        Dataframe containing metadata for the images and masks processed by imgtools autopipeline.
    readii_index : pd.DataFrame | None
        Dataframe containing metadata for the negative control images processed by make_negative_controls.py using READII
        If not supplied, will set up the index for the original images only.
    output_file_path : Path | None
        File path to save the FMCIB index csv out to. If not provided, will be set up as 
        `dirs.PROCDATA / f"{dataset_config['DATA_SOURCE']}_{dataset_name}" / "features" / f"fmcib_{dataset_name}_index.csv`

    Returns
    -------
    fmcib_index : pd.DataFrame
        Dataframe with columns:
        * SampleID: PatientID + SampleNumber from imgtools autopipeline
        * image_path - path to the image nifti file
        * Mask - path to the mask nifti file
        * DatasetName - name of the dataset these samples come from
        * SeriesInstanceUID_Image - Series ID from DICOM header for image
        * Modality_Image - image modality (e.g. CT, MR)
        * SeriesInstanceUID_Mask - Series ID from DICOM header for mask
        * Modality_Mask - mask modality (e.g. SEG, RTSTRUCT)
        * MaskID - name of region of interest (ROI) in the mask (e.g. GTVp)
        * readii_Permutation - permutation used for READII negative control
        * readii_Region - region used for READII negative control
        * coordX - global coordinates of the seed point around which features need to be extracted
        * coordY - global coordinates of the seed point around which features need to be extracted
        * coordZ - global coordinates of the seed point around which features need to be extracted
    """
    dataset_name = dataset_config['DATASET_NAME']

    # Use the pyradiomics index generator and then append the extra info for FMCIB    
    fmcib_index = generate_pyradiomics_index(dataset_config, mit_index, readii_index, output_file_path=dirs.PROCDATA / "temp" / f"temp_{dataset_name}_image_list.csv")

    # FMCIB expects a column named image_path, so prepend Image column with images dir path
    fmcib_index['image_path'] = fmcib_index.apply(lambda x: dirs.PROCDATA / f"{dataset_config['DATA_SOURCE']}_{dataset_name}" / "images" / x.Image, axis=1)

    # Append coordinates to the end of the index
    fmcib_index['coordX'] = 0
    fmcib_index['coordY'] = 0
    fmcib_index['coordZ'] = 0

    try:
        # If no output file path is provided, use default path setup, which makes a features directory for the dataset and saves the index there
        if output_file_path is None:
            output_file_path = dirs.PROCDATA / f"{dataset_config['DATA_SOURCE']}_{dataset_name}" / "features" / f"fmcib_{dataset_name}_index.csv"
        
        # Check that output file path is a .csv
        assert output_file_path.suffix == ".csv"

        # Create any missing parent directories for the output
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Save out the index file
        fmcib_index.to_csv(output_file_path, index=False)
    
    except AssertionError:
        message = f"output_file_path for generate_fmcib_index does not end in .csv. Path given: {output_file_path}"
        logger.error(message)
        raise

    return fmcib_index


@click.command()
@click.option('--dataset', type=click.STRING, help='Dataset configuration file name (e.g. NSCLC-Radiomics.yaml). Must be in config/datasets.')
@click.option('--method', type=click.STRING, default='pyradiomics', help='Feature extraction method to use.')
@click.option('--overwrite', type=click.BOOL, default=False, help='Overwrite existing index files.')
def generate_dataset_index(dataset: str, 
                           method: str = 'pyradiomics',
                           overwrite: bool = False
                           ) -> pd.DataFrame:
    """Create data index file for feature extraction listing image and mask file pairs.
    """
    if dataset is None:
        message = "Dataset name must be provided."
        logger.error(message)
        raise ValueError(message)

    # Load in dataset configuration settings from provided file
    config_dir_path = dirs.CONFIG / 'datasets'
    dataset_config = loadImageDatasetConfig(dataset, config_dir_path)

    dataset_name = dataset_config['DATASET_NAME']
    full_data_name = get_full_data_name(config_dir_path / dataset)
    
    # Construct image directory from DMP and config
    image_directory = dirs.PROCDATA / full_data_name / "images" 

    # Construct output file path from DMP and feature extraction type
    feature_extraction_type = method
    output_file_path = dirs.PROCDATA / full_data_name / "features" / feature_extraction_type / f"{feature_extraction_type}_{dataset_name}_index.csv"

    if output_file_path.exists() and not overwrite:
        message = f"{feature_extraction_type} index file already exists for {dataset_name}. Loading existing file."
        logger.info(message)
        try:
            dataset_index = pd.read_csv(output_file_path)
        except Exception as e:
            logger.error(f"Failed to load existing index file {output_file_path}. Consider using --overwrite to regenerate the index file.: {e}")
            raise
    else:
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Load imgtools autopipeline index file
    mit_index_path = image_directory / f'mit_{dataset_name}' / f'mit_{dataset_name}_index-simple.csv'
    if mit_index_path.exists():
        logger.info(f"Loading autopipeline dataset index file: {mit_index_path}")
        mit_index = pd.read_csv(mit_index_path)

        # Filter the mit_index by the modalities specified in the dataset config
        # Get just the rows with the desired image and mask modalities specified in the dataset config
        image_modality = dataset_config["MIT"]["MODALITIES"]["image"]
        mask_modality = dataset_config["MIT"]["MODALITIES"]["mask"]
        mit_index = get_masked_image_metadata(dataset_index = mit_index,
                                              dataset_config = dataset_config,
                                              image_modality = image_modality,
                                              mask_modality = mask_modality)
    else:
        logger.error(f"No existing index file found at {mit_index_path}. Run imgtools autopipeline first.")
        raise FileNotFoundError()

    # Check if any READII image processing settings have been specified
    regions, permutations, crop, resize = get_readii_settings(dataset_config)
    if (regions != [] and permutations) != [] or crop != '' or resize != []:
        try:
            # Load READII negative control index file generated by make_negative_controls.py if it exists
            readii_index_path = get_readii_index_filepath(dataset_config,
                                                          readii_image_dir = image_directory / f'readii_{dataset_name}')
            
            logger.info(f"Loading readii dataset index file: {readii_index_path}")
            readii_index = pd.read_csv(readii_index_path)

        except FileNotFoundError as e:
            logger.warning(f"No existing READII index file found for specified settings. No READII negative controls will be processed.")
            readii_index = None
    else:
        logger.info(f"No READII settings specified. Only MIT index will be used for extraction index generation.")
        readii_index = None

    match feature_extraction_type:
        case "pyradiomics":
            dataset_index = generate_pyradiomics_index(dataset_config,
                                                        mit_index,
                                                        readii_index,
                                                        output_file_path)
        case "fmcib":
            dataset_index = generate_fmcib_index(dataset_config,
                                                 mit_index,
                                                 readii_index,
                                                 output_file_path)
        case _:
            message = f"Index generator doesn't exist for {feature_extraction_type}."
            logger.debug(message)
            raise NotImplementedError(message)
    
    return dataset_index



if __name__ == "__main__":
    generate_dataset_index()
    


