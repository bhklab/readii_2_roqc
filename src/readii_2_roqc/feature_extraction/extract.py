import logging
from collections import OrderedDict
from pathlib import Path
from typing import Generator

import click
import pandas as pd
import SimpleITK as sitk
from damply import dirs
from joblib import Parallel, delayed
from radiomics import featureextractor, setVerbosity

from tqdm import tqdm

from readii_2_roqc.utils.loaders import load_dataset_config
from readii_2_roqc.utils.metadata import remove_slice_index_from_string
from readii_2_roqc.utils.settings import get_extraction_index_filepath

logger = logging.getLogger(__name__)

def sample_feature_writer(feature_vector : OrderedDict,
                          metadata : dict[str, str],
                          extraction_method: str,
                          extraction_settings_name : str
                          ) -> OrderedDict[str, str]:
    """Write out the feature vector and metadata to a csv file.
    
    Parameters
    ----------
    feature_vector : OrderedDict
        Dictionary of features extracted for the set of samples listed in metadata using the provided extraction method and settings.
    metadata : dict[str, str]
        Dictionary of metadata for each of the samples in feature_vector. At minimum, should include a SampleID.
    extraction_method: str
        Method of feature extraction being performed. Used to construct output path.
    extraction_settings_name : str
        Name of the settings used for feature extraction. Used to construct output path.
    
    Returns
    -------
        OrderedDict[str, str]
            Combined metadata and features that was saved out.
    """

    resize = metadata['readii_Resize']
    if resize == '':
        image_size_str = "original"
    else:
        # Get image size data for output path
        image_size_str = remove_slice_index_from_string(resize)
    
    # Construct output path with elements from metadata
    output_path = dirs.PROCDATA / f"{metadata['DataSource']}_{metadata['DatasetName']}" / "features" / extraction_method / image_size_str / extraction_settings_name / metadata['SampleID'] / metadata['MaskID'] / f"{metadata['readii_Permutation']}_{metadata['readii_Region']}_features.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Set up metadata as an OrderedDict to be combined with the features
    if not isinstance(metadata, OrderedDict):
        metadata = metadata_setup(metadata)
    
    # Combine metadata and feature vector
    metadata.update(feature_vector)
    
    # Save out the metadata to a csv
    with output_path.open('w') as f:
        f.writelines(f"{key};{value}\n" for key, value in metadata.items())

    logger.info(f"Features written to {output_path}.")
    return metadata



def metadata_setup(metadata : dict[str, str] | pd.Series) -> OrderedDict:
    """Convert metadata to an OrderedDict.
    
    Parameters
    ----------
    metadata : dict[str, str] | pd.Series
        Metadata to converted into an OrderedDict. Can be a dict or pd.Series.
    
    Returns
    -------
    od_metadata : OrderedDict
        Metadata as an OrderedDict object.
    
    Raises
    ------
    NotImplementedError
        If metadata is not of type dict or pd.Series
    """
    match metadata:
        case dict():
            od_metadata = OrderedDict(metadata)
        case pd.Series():
            od_metadata = metadata.to_dict(into=OrderedDict)
        case _:
            message = f"Metadata must be of dict or pd.Series type. Other types not handled yet. Metadata you've passed is of type {type(metadata)}."
            logger.debug(message)
            raise NotImplementedError(message)

    return od_metadata


def pyradiomics_extract(settings: Path | str,
                        image: sitk.Image,
                        mask: sitk.Image,
                        metadata : OrderedDict | dict[str, str] | pd.Series | None = None,
                        overwrite : bool = False
                        ) -> pd.Series:
    """Extract PyRadiomic features from an image and mask pair based on configuration from settings file.
    
    Parameters
    ----------
    
    Returns
    -------

    """
    # Set PyRadiomics verbosity to critical only
    setVerbosity(50)

    if metadata is None:
        message = "`metadata` must be provided when overwrite is False."
        raise ValueError(message)
    
    resize = metadata['readii_Resize']
    if resize == '':
        image_size_str = "original"
    else:
        # Get image size data for output path
        image_size_str = remove_slice_index_from_string(resize)

    # Check if feature file already exists and if overwrite is specified
    sample_feature_file_path = dirs.PROCDATA / f"{metadata['DataSource']}_{metadata['DatasetName']}" / "features" / "pyradiomics" / image_size_str / Path(settings).stem / metadata['SampleID'] / metadata['MaskID'] / f"{metadata['readii_Permutation']}_{metadata['readii_Region']}_features.csv"
    if sample_feature_file_path.exists() and not overwrite:
        logger.info(f"Features for {metadata['SampleID']} {metadata['readii_Permutation']} {metadata['readii_Region']} {metadata['Modality_Image']} image and {metadata['MaskID']} mask have already been extracted.")
        # Load the existing feature file
        sample_feature_df = pd.read_csv(sample_feature_file_path, index_col=0, header=None, sep=";")
        # collapse the single data column into a flat OrderedDict  
        sample_feature_vector = OrderedDict(sample_feature_df.iloc[:, 0].to_dict())  

    else:
        # Confirm settings file exists
        if not Path(settings).exists():
            message = f"Settings file for PyRadiomics feature extraction at {settings} does not exist."
            logger.error(message)
            raise FileNotFoundError(message)

        # Convert settings Path to string for pyradiomics to read it
        if isinstance(settings, Path):
            settings = str(settings)

        try:
            logger.info(f"Extraction features for {metadata['SampleID']} {metadata['readii_Permutation']} {metadata['readii_Region']} {metadata['Modality_Image']} image and {metadata['MaskID']} mask")
 
            # Set up PyRadiomics feature extractor with provided settings file (expects a string, not a pathlib Path)
            extractor = featureextractor.RadiomicsFeatureExtractor(settings)

            sample_feature_vector = extractor.execute(image, mask)

        except Exception as e:
            logger.debug(f"Feature extraction failed for this sample: {e}")

            sample_feature_vector = OrderedDict()
        
        if len(sample_feature_vector) > 0:
            logger.info("Writing out extracted features.")
            # Save out the feature vector with the metadata appended to the front
            sample_feature_writer(feature_vector=sample_feature_vector,
                                metadata=metadata,
                                extraction_method="pyradiomics",
                                extraction_settings_name=Path(settings).stem)

    # Returning this vector of features on its own with no metadata on the front
    return sample_feature_vector



def extract_sample_features(sample_data: pd.Series,
                            method: str,
                            settings: Path | str,
                            overwrite: bool = False) -> OrderedDict:
    """Extract features from a single sample using the specified method and settings.

    Parameters
    ----------
    sample_data : pd.Series
        The input data for the sample. This will be treated as metadata for the sample and appended to the front of the feature vector when it gets saved out.
        It should contain keys like 'DatasetName', 'Image', 'Mask', etc.
        The 'Image' and 'Mask' keys should point to the image and mask files respectively
        within the dataset's image directory.
        The 'DatasetName' key should point to the name of the dataset.
        Example:
        ```
        {   'DatasetName': 'NSCLC-Radiomics_test',
            'DataSource': 'TCIA',
            'Image': 'image.nii.gz',
            'Mask': 'mask.nii.gz',
            'SampleID': 'sample_001',
            'MaskID': 'GTV',
            'readii_Permutation': 'original'
        }
        ```
    method : str
        The feature extraction method to use.
    settings : str
        Name of the settings file for the feature extraction method. Should be in the config/<method> directory.
    overwrite : bool
        Whether to overwrite existing feature files.

    Returns
    -------
    OrderedDict
        The extracted features for the sample. No metadata will be prepended to this vector.
    """
    # Set up settings file path for the feature extraction method
    settings_path = dirs.CONFIG / method / settings
    if not settings_path.exists():
        message = f"Settings file for {method} feature extraction does not exist at {settings_path}."
        logger.error(message)
        raise FileNotFoundError(message)
    
    data_dir = dirs.PROCDATA / f"{sample_data['DataSource']}_{sample_data['DatasetName']}" / "images"

    image = sitk.ReadImage(data_dir / sample_data['Image'])
    mask = sitk.ReadImage(data_dir / sample_data['Mask'])

    match method:
        case "pyradiomics":
            # Extract features using PyRadiomics
            sample_feature_vector = pyradiomics_extract(settings=settings_path,
                                                        image=image,
                                                        mask=mask,
                                                        metadata=sample_data,
                                                        overwrite=overwrite)
        case _:
            message = f"Feature extraction method {method} is not implemented yet."
            logger.error(message)
            raise NotImplementedError(message)
    
    return sample_feature_vector



def compile_dataset_features(dataset_index: pd.DataFrame,
                             method: str,
                             settings_name: str,
                             overwrite: bool = False 
                             ) -> dict[str, pd.DataFrame]:
    """Compile features from all samples of each image type in a directory into a single DataFrame and save it to a CSV file.
    
    Parameters
    ----------
    dataset_index : pd.DataFrame
        The dataset index DataFrame containing metadata for each sample.
        It should contain columns like 'DataSource', 'DatasetName', 'readii_Permutation', 'readii_Region', etc.
    method : str
        The feature extraction method used to extract the features. This will be used to construct the output directory structure.
    settings_name : str
        The name of the settings file used for feature extraction. This will be used to construct the output directory structure.
    overwrite : bool = False
        Whether to overwrite existing compiled dataset files.
    Returns
    -------
    compiled_dataset_features : dict[str, pd.DataFrame]
        A dictionary where keys are image type identifiers (e.g., "original_full", "original_partial") and values are DataFrames containing the compiled features for each image type.
    """
    dataset_row = dataset_index.iloc[0]
    # Validate that all rows share the same DataSource and DatasetName  
    if not (dataset_index['DataSource'].nunique() == 1 and dataset_index['DatasetName'].nunique() == 1):  
        message = "Dataset index contains mixed DataSource or DatasetName values."  
        logger.error(message)  
        raise ValueError(message)
    
    resize = dataset_row['readii_Resize']
    if resize == '':
        image_size_str = 'original'
    else:
        image_size_str = remove_slice_index_from_string(resize)

    # Set up the directory structure for the features in the processed (samples) and results (datasets) directories
    features_dir_struct = Path(f"{dataset_row['DataSource']}_{dataset_row['DatasetName']}") / "features" / method / image_size_str / settings_name

    # Set up path to the directory containing the sample feature files
    sample_features_dir = dirs.PROCDATA / features_dir_struct

    # Get each of the image types in the dataset index as a set of tuples
    readii_image_classes = {image_class for image_class in dataset_index[['readii_Permutation', 'readii_Region']].itertuples(index=False, name=None)}

    # Check for existing result feature dataset files
    existing_dataset_files = sorted((dirs.RESULTS / features_dir_struct).glob('*.csv'))
    # Initialize dictionary to store compiled feature dataframes
    compiled_dataset_features: dict[str, pd.DataFrame] = {}
    if existing_dataset_files and not overwrite:
        # Get the image classes in the same format as readii_image_classes (a set of tuples)
        compiled_image_classes = {tuple(file.name.removesuffix('_features.csv').split('_', 1)) for file in existing_dataset_files}
        
        # Check whether there are new image classes to compile
        if readii_image_classes.issubset(compiled_image_classes):
            message = "All requested feature sets have already been generated for these samples and compiled into results for this dataset. Set overwrite to True if you want to re-process these."
            logger.info(message)
            
            # Load in the existing compiled dataset files into a dictionary to match function output
            compiled_dataset_features = {file.name.removesuffix('_features.csv'):pd.read_csv(file) for file in existing_dataset_files}
            return compiled_dataset_features
        else:
            message = "Some requested feature sets have already been compiled. These will not be rerun, but loaded from existing files. Set overwrite to True if you want to re-compile all image type feature sets."
            logger.info(message)

    else:
        for permutation, region in readii_image_classes:
            logger.info(f"Compiling features for {permutation} {region} images.")
            # Filter the dataset index for this image class
            filtered_class_index = dataset_index[(dataset_index['readii_Permutation'] == permutation) & 
                                        (dataset_index['readii_Region'] == region)]

            # If there are no samples for this image class, skip it
            if filtered_class_index.empty:
                logger.warning(f"No samples found for image class {permutation} {region}. Skipping.")
                continue

            # Compile features for this image class
            # Regex for directory search
            filename_pattern = f"**/{permutation}_{region}_features.csv"
            # Recursively search for sample feature files for this image type and sort them into a list
            sample_feature_files = sorted(sample_features_dir.rglob(filename_pattern))

            # Set up output file path for this image type
            dataset_features_path = dirs.RESULTS / features_dir_struct / f"{permutation}_{region}_features.csv"
            
            # Check if this image type has compiled features already 
            if dataset_features_path.exists() and not overwrite:
                # Load existing features into dictionary for return
                compiled_dataset_features[f"{permutation}_{region}"] = pd.read_csv(dataset_features_path)

            # No existing feature file OR overwrite requested
            else:
                dataset_features_path.parent.mkdir(parents=True, exist_ok=True)

                # Generator
                def non_empty_dfs(file_list:list[Path]) -> Generator[pd.DataFrame, list[Path], None]:
                    for file in file_list:
                        try:
                            file_df = pd.read_csv(file, index_col=0, header=None, sep=";")
                            if not file_df.empty:
                                yield file_df.T
                        except pd.errors.EmptyDataError:
                            pass

                try:
                    # Find all non-empty feature dataframes in the globbed list and concatenate them
                    dataset_features = pd.concat(non_empty_dfs(sample_feature_files))
                    # Sort the dataframes by the sample ID column
                    dataset_features = dataset_features.sort_values(by="SampleID")
                    # Save out the combined feature dataframe
                    dataset_features.to_csv(dataset_features_path, index=False)

                except ValueError:
                    # Handle case where all dataframes are empty
                    logger.error(f"No non-empty dataframes found for {permutation} {region}.")
                    # Create empty dataframe for compiled dataset features
                    dataset_features = pd.DataFrame()
                    # write empty file to the output file
                    with dataset_features_path.open("w") as f:
                        # write an empty file
                        f.write("")
                    logger.error(f"Empty file written to {dataset_features_path}")

                compiled_dataset_features[f"{permutation}_{region}"] = dataset_features

    return compiled_dataset_features



@click.command()
@click.argument('dataset', type=click.STRING)
@click.argument('method', type=click.Choice(['pyradiomics']))
@click.argument('settings', type=click.STRING)
@click.option('--overwrite', type=click.BOOL, default=False, help='Overwrite existing feature files.')
@click.option('--parallel', type=click.BOOL, default=False, help='Run feature extraction in parallel.')
@click.option('--jobs', type=click.INT, help="Number of jobs to give parallel processor", default=-1)
def extract_dataset_features(dataset: str,
                             method: str,
                             settings: str | Path,
                             overwrite: bool = False,
                             parallel: bool = False,
                             jobs: int = -1) -> dict[str, pd.DataFrame]: 
    """Extract features from a dataset using the specified method and settings.

    Parameters
    ----------
    dataset : str
        Name of the dataset to perform extraction on. Must have a configuration file in the config/datasets directory.
    method : str
        Feature extraction method to use.
    settings : str | Path
        Path to the feature extraction settings file.
    overwrite : bool = False
        Whether to overwrite existing feature files.
    parallel : bool = False
        Whether to run feature extraction in parallel. Defaults to False.
    jobs : int = -1
        Number of jobs to give parallel processor.

    Returns
    -------
    dict[str, pd.DataFrame]
        Compiled feature tables per image class keyed by "<permutation>_<region>".
    """

    dirs.LOGS.mkdir(parents=True, exist_ok=True)  
    logging.basicConfig(  
        filename=str(dirs.LOGS / f"{dataset}_extract.log"),  
        encoding='utf-8',  
        level=logging.DEBUG,  
        force=True  
    )

    if dataset is None:
        message = "Dataset name must be provided."
        logger.error(message)
        raise ValueError(message)
    
    # Load in dataset configuration settings from provided dataset name
    dataset_config, dataset_name, full_data_name = load_dataset_config(dataset)
    logger.info(f"Extraction {method} radiomic features for {dataset_name}")

    try:
        # Load the dataset index
        dataset_index_path = get_extraction_index_filepath(dataset_config,
                                                           extract_features_dir = dirs.PROCDATA / full_data_name / "features" / method)
        dataset_index = pd.read_csv(dataset_index_path, keep_default_na=False)
    except FileNotFoundError:
        logger.error(f"Dataset index file for {method} feature extraction not found for {full_data_name}.")
        raise

    # Add dataset source to metadata for file loading and saving purposes
    if 'DataSource' not in dataset_index.columns:
        dataset_index['DataSource'] = dataset_config['DATA_SOURCE']

    # PyRadiomics READII does not support crop in this pipeline; raise if any crop value is present
    if method == 'pyradiomics':
        crop_series = dataset_index['readii_Crop']
        has_crop = crop_series.notna() & (crop_series.astype(str).str.strip() != '')
        if has_crop.any():
            message = 'No crop methods have been implemented for PyRadiomics extraction with READII yet.'
            logger.error(message)
            raise NotImplementedError(message)

    logger.info("Starting feature extraction for individual image type + mask pairs.")
    # Extract features for each sample in the dataset index
    if parallel:
        # Use joblib to parallelize feature extraction
        Parallel(n_jobs=jobs)(
            delayed(extract_sample_features)(
                sample_data=sample_data,
                method=method,
                settings=settings,
                overwrite=overwrite
            )
            for _, sample_data in tqdm(
                dataset_index.iterrows(),
                desc=f"Extracting {method} features",
                total=len(dataset_index)
            )
        )
    else:
        # Sequentially extract features
        for _, sample_data in tqdm(
            dataset_index.iterrows(),
            desc=f"Extracting {method} features",
            total=len(dataset_index)
        ):
            extract_sample_features(
                sample_data=sample_data,
                method=method,
                settings=settings,
                overwrite=overwrite
            )

    logger.info("Compiling sample feature vectors into complete dataset table.")
    # Collect all the sample feature vectors for the dataset into a DataFrame for each image type
    dataset_feature_vectors = compile_dataset_features(dataset_index,
                                                       method,
                                                       settings_name=Path(settings).stem,
                                                       overwrite = overwrite)
    
    return dataset_feature_vectors



if __name__ == "__main__":
    extract_dataset_features()