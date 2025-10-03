from pathlib import Path

import click
import pandas as pd
from damply import dirs
from readii.io.loaders import loadImageDatasetConfig
from readii.process.config import get_full_data_name
from readii.utils import logger

from readii_2_roqc.utils.metadata import get_masked_image_metadata, make_edges_df, remove_slice_index_from_string
from readii_2_roqc.utils.settings import get_readii_settings, get_resize_string, get_readii_index_filepath

def get_mit_extraction_index(dataset_config: dict,
                             mit_index_path: Path):
    """Set up med-imagetools index dataframe for feature extraction.
    
    Parameters
    ----------
    dataset_config : dict
        Configuration settings for a dataset, loaded with loadImageDatasetConfig
    mit_index_path : Path
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

    if mit_index_path.exists():
        logger.info(f"Loading autopipeline dataset index file: {mit_index_path}")
        mit_index = pd.read_csv(mit_index_path)
    else:
        message = f"No existing index file found at {mit_index_path}. Run imgtools autopipeline first."
        logger.error(message)
        raise FileNotFoundError(message)


    # Filter the mit_index by the modalities specified in the dataset config
    # Get just the rows with the desired image and mask modalities specified in the dataset config
    image_modality = dataset_config["MIT"]["MODALITIES"]["image"]
    mask_modality = dataset_config["MIT"]["MODALITIES"]["mask"]
    mit_index = get_masked_image_metadata(dataset_index = mit_index,
                                            dataset_config = dataset_config,
                                            image_modality = image_modality,
                                            mask_modality = mask_modality)

    # Get single row for each image and mask pair
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
                              "readii_Region": "full",
                              "readii_Crop": '',
                              "readii_Resize": mit_edges_index.apply(lambda x: x['size_image'].replace(', ', "_").strip('()'), axis=1)
                             }
                       )



def get_readii_extraction_index(dataset_config: dict,
                                readii_index_path: Path):
    """Set up readii index dataframe for feature extraction on READII processed images using the index file generated from negative control generation.
    
    Parameters
    ----------
    dataset_config : dict
        Configuration settings for a dataset, loaded with loadImageDatasetConfig
    readii_index : pd.DataFrame
        Dataframe containing metadata for the images and masks processed by READII make_negative_controls
    
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

    if readii_index_path.exists():
        logger.info(f"Loading autopipeline dataset index file: {readii_index_path}")
        readii_index = pd.read_csv(readii_index_path)
    else:
        message = f"No existing index file found at {readii_index_path}. Run readii_negative to get processed images for feature extraction."
        logger.error(message)
        raise FileNotFoundError(message)

    # Load the requested image processing settings from configuration
    regions, permutations, crop, resize = get_readii_settings(dataset_config)
    
    # Add the original full negative control options to their respective list to catch these when a crop has been applied
    if crop is not None:
        regions += ["full"]
        permutations += ["original"]

    # Filter the index file by the specified READII settings
    settings_readii_index = readii_index[readii_index['Region'].isin(regions) & readii_index['Permutation'].isin(permutations)]

    image_modality = dataset_config["MIT"]["MODALITIES"]["image"]
    mask_modality = dataset_config["MIT"]["MODALITIES"]["mask"]
    
    return pd.DataFrame(data={"SampleID": settings_readii_index.SampleID,
                              "Image": settings_readii_index.apply(lambda x: f"{Path(f'readii_{dataset_name}') / x.filepath}", axis=1),
                              "Mask": settings_readii_index.apply(lambda x: f"{Path(f'mit_{dataset_name}') / x.SampleID / f'{x.MaskID}.nii.gz'}", axis=1),
                              "DatasetName": dataset_name,
                              "SeriesInstanceUID_Image": "",
                              "Modality_Image": image_modality,
                              "SeriesInstanceUID_Mask": "",
                              "Modality_Mask": mask_modality,
                              "MaskID": settings_readii_index.apply(lambda x: f"{Path(x.MaskID).name}", axis=1),
                              "readii_Permutation": settings_readii_index["Permutation"],
                              "readii_Region": settings_readii_index["Region"],
                              "readii_Crop": settings_readii_index["crop"],
                              "readii_Resize": settings_readii_index["Resize"]
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

    if readii_index is not None:
        # Concatenate the original and negative control image index dataframes
        pyradiomics_index = pd.concat([mit_index, readii_index], ignore_index=True, axis=0)
        # Sort the resulting index by negative control settings, then SampleID and MaskID
        pyradiomics_index = pyradiomics_index.sort_values(by=['readii_Permutation', 'readii_Region', 'SampleID', 'MaskID'], ignore_index=True)

    else:
        # No negative control images to process, just use original images index
        pyradiomics_index = mit_index

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
                         readii_index: pd.DataFrame | None = None,
                         output_file_path: Path | None = None
                        ) -> pd.DataFrame:
    """Set up and save out index file for Foundation Model for Cancer Image Biomarkers (FMCIB) feature extraction. Output file contains columns for "image_path", "coordX", "coordY", "coordZ".
    
    Parameters
    ----------
    dataset_config : dict
        Configuration settings for a dataset, loaded with loadImageDatasetConfig
    readii_index : pd.DataFrame | None
        Dataframe containing metadata for the images cropped and/or negative control generated by make_negative_controls.py using READII
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

    # Use the readii extraction index generator since FMCIB needs cropped images that would've been processed by READII   
    fmcib_index = readii_index.sort_values(by=['readii_Crop', 'readii_Permutation', 'readii_Region', 'SampleID', 'MaskID'], ignore_index=True)

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
@click.argument('dataset', type=click.STRING)
@click.argument('method', type=click.Choice(['pyradiomics', 'fmcib']))
@click.option('--overwrite', type=click.BOOL, default=False, help='Overwrite existing index files.')
def generate_dataset_index(dataset: str, 
                           method: str = 'pyradiomics',
                           overwrite: bool = False
                           ) -> pd.DataFrame:
    """Create data index file for feature extraction listing image and mask file pairs.

    Parameters
    ----------
    dataset:str
        Dataset name (e.g. NSCLC-Radiomics). Must have a .yaml file in config/datasets.
    method:str
        Feature extraction method to generate index for. Options are pyradiomics and fmcib.
    overwrite:bool = False
        Whether to overwrite existing index files. Default is False.
    Returns
    -------
    dataset_index:pd.DataFrame
        Dataframe listing metadata required for specified method's feature extraction process.
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

    # Load imgtools autopipeline index file
    mit_index_path = image_directory / f'mit_{dataset_name}' / f'mit_{dataset_name}_index-simple.csv'
    mit_index = get_mit_extraction_index(dataset_config=dataset_config,
                                         mit_index_path=mit_index_path)

    # Check if any READII image processing settings have been specified
    regions, permutations, crop, resize = get_readii_settings(dataset_config)
    print(regions)
    print(permutations)
    print(crop)
    print(resize)
    if (regions is not None and permutations is not None) or crop is not None or resize is not None:
        # Load READII negative control index file generated by make_negative_controls.py if it exists
        readii_index_path = get_readii_index_filepath(dataset_config,
                                                      readii_image_dir = image_directory / f'readii_{dataset_name}')
        
        readii_index = get_readii_extraction_index(dataset_config=dataset_config,
                                                    readii_index_path=readii_index_path)

    else:
        logger.info(f"No READII settings specified. Only MIT index will be used for extraction index generation.")
        readii_index = None

    # Construct output file path from DMP and feature extraction type
    feature_extraction_type = method
    if crop is None and resize is None:
        # Get the x and y dimension of the image and put this with an n as the size value for the output folder
        image_type = remove_slice_index_from_string(img_size=mit_index.readii_Resize[0])
    else:
        image_type = f'{crop}_{get_resize_string(resize)}'

    output_file_path = dirs.PROCDATA / full_data_name / "features" / feature_extraction_type / image_type / f"{feature_extraction_type}_{dataset_name}_index.csv"

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

    match feature_extraction_type:
        case "pyradiomics":
            dataset_index = generate_pyradiomics_index(dataset_config,
                                                        mit_index,
                                                        readii_index,
                                                        output_file_path)
        case "fmcib":
            dataset_index = generate_fmcib_index(dataset_config,
                                                 readii_index,
                                                 output_file_path)
        case _:
            message = f"Index generator doesn't exist for {feature_extraction_type}."
            logger.debug(message)
            raise NotImplementedError(message)
    
    return dataset_index



if __name__ == "__main__":
    generate_dataset_index()
    


