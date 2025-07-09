from importlib import metadata
from itertools import product
from pathlib import Path
import SimpleITK as sitk
import pandas as pd
from collections import OrderedDict
import click
from joblib import Parallel, delayed

from radiomics import featureextractor, setVerbosity
from readii.utils import logger
from readii.io.loaders import loadImageDatasetConfig
from readii.process.config import get_full_data_name

from damply import dirs

def sample_feature_writer(feature_vector : OrderedDict,
                          metadata : dict[str, str],
                          extraction_method: str,
                          extraction_settings_name : str):
    
    """Write out the feature vector and metadata to a csv file."""
    # Construct output path with elements from metadata
    output_path = dirs.PROCDATA / f"{metadata['DataSource']}_{metadata['DatasetName']}" / "features" / extraction_method / extraction_settings_name / metadata['SampleID'] / metadata['MaskID'] / f"{metadata['readii_Permutation']}_{metadata['readii_Region']}_features.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Set up metadata as an OrderedDict to be combined with the features
    if not isinstance(metadata, OrderedDict):
        metadata = metadata_setup(metadata)
    
    # Combine metadata and feature vector
    metadata.update(feature_vector)
    
    # Save out the metadata to a csv
    with open(output_path, 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key};{value}\n")

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
            od_metdata = OrderedDict(metadata)
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

    # TODO: add check if feature file already exists and overwrite if specified

     # Confirm settings file exists
    try:
        assert Path(settings).exists()
    except AssertionError as e:
        logger.error(f"Settings file for PyRadiomics feature extraction at {settings} does not exist.")
        raise e

    # Convert settings Path to string for pyradiomics to read it
    if isinstance(settings, Path):
        settings = str(settings)

    try:
        # Set up PyRadiomics feature extractor with provided settings file (expects a string, not a pathlib Path)
        extractor = featureextractor.RadiomicsFeatureExtractor(settings)

        sample_feature_vector = extractor.execute(image, mask)

    except Exception as e:
        print(f"Feature extraction failed for this sample: {e}")

    # Save out the feature vector with the metadata appended to the front
    output_path = sample_feature_writer(feature_vector = sample_feature_vector, 
                                        metadata = metadata,
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
    try:
        # Load the settings file for the feature extraction method
        settings_path = dirs.CONFIG / method / settings
        assert settings_path.exists()
    except AssertionError as e:
        logger.error(f"Settings file for {method} feature extraction does not exist at {settings_path}.")
        raise e
    
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
                             settings_name: str 
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

    Returns
    -------
    compiled_dataset_features : dict[str, pd.DataFrame]
        A dictionary where keys are image type identifiers (e.g., "original_full", "original_partial") and values are DataFrames containing the compiled features for each image type.
    """
    # Set up the directory structure for the features in the processed (samples) and results (datasets) directories
    features_dir_struct = Path(f"{dataset_index.iloc[0]['DataSource']}_{dataset_index.iloc[0]['DatasetName']}") / "features" / method / settings_name

    # Set up path to the directory containing the sample feature files
    sample_features_dir = dirs.PROCDATA / features_dir_struct

    # Get each of the image types in the dataset index
    readii_image_classes = set(product(dataset_index['readii_Permutation'].unique(), dataset_index['readii_Region'].unique()))

    # Initialize dictionary to store compiled feature dataframes
    compiled_dataset_features = {}

    for permutation, region in readii_image_classes:
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
        dataset_features_path.parent.mkdir(parents=True, exist_ok=True)

        # Generator
        empty_files = [] 
        def non_empty_dfs(file_list):
            for file in file_list:
                try:
                    df = pd.read_csv(file, index_col=0, header=None, sep=";")
                    if not df.empty:
                        empty_files.append(file)
                        yield df.T
                except pd.errors.EmptyDataError:
                    pass

        try:
            # Find all non-empty feature dataframes in the globbed list and concatenate them
            dataset_features = pd.concat(non_empty_dfs(sample_feature_files))
            # Sort the dataframes by the sample ID column
            dataset_features.sort_values(by="SampleID", inplace=True)
            # Save out the combined feature dataframe
            dataset_features.to_csv(dataset_features_path, index=False)

        except ValueError:
            # Handle case where all dataframes are empty
            logger.error(f"No non-empty dataframes found for {permutation} {region}.")
            # write empty file to the output file
            with open(dataset_features_path, "w") as f:
                # write an empty file
                f.write("")
            logger.error(f"Empty file written to {dataset_features_path}")

        compiled_dataset_features[f"{permutation}_{region}"] = dataset_features

    return compiled_dataset_features



@click.command()
@click.option('--dataset', type=click.STRING, required=True, help='Name of the dataset to perform extraction on.')
@click.option('--method', type=click.Choice(['pyradiomics']), required=True, help='Feature extraction method to use.')
@click.option('--settings', type=click.STRING, required=True, help='Name of the feature extraction settings file in config/<method>.')
@click.option('--overwrite', is_flag=True, default=False, help='Overwrite existing feature files.')
@click.option('--parallel', is_flag=True, default=False, help='Run feature extraction in parallel.')
def extract_dataset_features(dataset: str,
                             method: str,
                             settings: str | Path,
                             overwrite: bool = False,
                             parallel: bool = False) -> Path:
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

    Returns
    -------
    Path
        Path to the output feature file.
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
    logger.info(f"Extraction {method} radiomic features for {dataset_name}")

    try:
        # Load the dataset index
        dataset_index_path = dirs.PROCDATA / full_data_name / "features" / method / f"{method}_{dataset_name}_index.csv"
        dataset_index = pd.read_csv(dataset_index_path)
    except FileNotFoundError as e:
        logger.error(f"Dataset index file for {method} feature extraction does not exist at {dataset_index_path}.")
        raise e

    # Add dataset source to metadata for file loading and saving purposes
    if 'DataSource' not in dataset_index.columns:
        dataset_index['DataSource'] = dataset_config['DATA_SOURCE']

    # Extract features for each sample in the dataset index
    if parallel:
        # Use joblib to parallelize feature extraction
        feature_vectors = Parallel(n_jobs=-1)(
            delayed(extract_sample_features)(sample_data = sample_data, 
                                             method = method, 
                                             settings = settings, 
                                             overwrite = overwrite)
            for _, sample_data in dataset_index.iterrows()
        )
    else:
        # Sequentially extract features
        feature_vectors = [
            extract_sample_features(sample_data = sample_data, 
                                    method = method, 
                                    settings = settings, 
                                    overwrite = overwrite)
            for _, sample_data in dataset_index.iterrows()
        ]

    # Collect all the sample feature vectors for the dataset into a DataFrame for each image type
    dataset_feature_vectors = compile_dataset_features(dataset_index,
                                                       method,
                                                       settings_name=Path(settings).stem)
    
    return dataset_feature_vectors



if __name__ == "__main__":
    extract_dataset_features()