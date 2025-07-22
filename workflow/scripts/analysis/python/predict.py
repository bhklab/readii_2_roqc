from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from damply import dirs
import click
from readii.utils import logger
from readii.io.loaders import loadImageDatasetConfig, loadFileToDataFrame
from readii.process.config import get_full_data_name
from readii.process.label import (
    eventOutcomeColumnSetup,
    getPatientIdentifierLabel,
    timeOutcomeColumnSetup,
)
from readii.process.split import splitDataByColumnValue
from readii.process.subset import selectByColumnValue

# Load signature file

def load_signature_config(file: str | Path) -> pd.Series:
    """Load in a predictive signature from a yaml file. The signature is expected to be organized as a dictionary of {feature: value} pairs. Features should be strings and values a numeric type.

    Parameters
    ----------
    file : str | Path
        Name of the signature file contained in the config/signatures directory.

    Returns
    -------
    signature : pd.Series
        Signature set up as a pd.Series, with the index being the features.

    Raises
    ------
    ValueError
        If the input file does not end in ".yaml"
    """

    signature_file_path = dirs.CONFIG / "signatures" / file

    if signature_file_path.suffix not in [".yaml", ".yml"]:
        logger.error(f"Signature file must be a .yaml. Provided file is type {signature_file_path.suffix}")
        raise ValueError

    try:
        with signature_file_path.open('r') as f:
            yaml_data = yaml.safe_load(f)
            if not isinstance(yaml_data, dict):
                raise TypeError("ROI match YAML must contain a dictionary")
            signature = pd.Series(yaml_data['signature'])
    except Exception as e:
        message = f"Error loading YAML file: {e}"
        logger.error(message)
        raise

    return signature



def prediction_data_splitting(dataset_config,
                              data : pd.DataFrame):
    """Split metadata into train and test for model development and validation purposes"""
    split_settings = dataset_config['ANALYSIS']['TRAIN_TEST_SPLIT']
    if split_settings['split']:
        split_data = splitDataByColumnValue(data,
                                            split_col_data=split_settings['split_variable'],
                                            impute_value=split_settings['impute'])

    return split_data


def insert_mit_index(dataset_config: str,
                     data_to_index: pd.DataFrame
                     ) -> pd.DataFrame:
    """Add the Med-ImageTools SampleID index column to a dataframe (e.g. a clinical table) to align with processed imaging data"""
    # Find the existing patient identifier for the data_to_index
    existing_pat_id = getPatientIdentifierLabel(data_to_index)

    # Load the med-imagetools autopipeline simple index output for the dataset
    full_dataset_name = f"{dataset_config['DATA_SOURCE']}_{dataset_config['DATASET_NAME']}"
    mit_index_path = dirs.PROCDATA / full_dataset_name / "images" / f"mit_{dataset_config['DATASET_NAME']}" / f"mit_{dataset_config['DATASET_NAME']}_index-simple.csv"
    
    if mit_index_path.exists():
        mit_index = loadFileToDataFrame(mit_index_path)
    else:
        message = f"Med-ImageTools autopipeline index simple output don't exist for the {full_dataset_name} dataset. Run autopipeline to generate this file."
        print(message)
        logger.error(message)
        raise FileNotFoundError(message)

    # Generate a mapping from PatientID to SampleID (PatientID_SampleNumber from Med-ImageTools autopipeline output)
    id_map = mit_index["PatientID"].astype(str) + "_" + mit_index['SampleNumber'].astype(str).str.zfill(4)
    id_map.index = mit_index["PatientID"]
    id_map = id_map.drop_duplicates()

    # Apply the map to the dataset to index
    data_to_index['SampleID'] = data_to_index[existing_pat_id].map(id_map)

    return data_to_index


def clinical_data_setup(dataset_config,
                       full_dataset_name : str | None = None
                       ) -> pd.DataFrame:
    """Process the clinical data to get outcome variables for use in signature prediction"""
    if full_dataset_name is None:
        full_dataset_name = f"{dataset_config['DATA_SOURCE']}_{dataset_config['DATASET_NAME']}"

    # load clinical metadata
    clinical = dataset_config['CLINICAL']
    clinical_path = dirs.RAWDATA / full_dataset_name / "clinical" / clinical['FILE']
    clinical_data = loadFileToDataFrame(clinical_path)

    # insert the MIT index
    clinical_data = insert_mit_index(dataset_config, clinical_data)

    # Set the MIT SampleIDs as the index for clinical data
    clinical_data = clinical_data.set_index('SampleID')

    # Drop rows based on exclusion variables in config file
    if len(clinical['EXCLUSION_VARIABLES']) != 0:
        clinical_data = selectByColumnValue(clinical_data,
                                            exclude_col_values = clinical['EXCLUSION_VARIABLES'])

    return clinical_data


def outcome_data_setup(dataset_config,
                       dataframe_with_outcome: pd.DataFrame,
                       standard_event_label : str = "survival_event_binary",
                       standard_time_label : str = "survival_time_years"
                       ) -> pd.DataFrame:
    """Set up survival time in years and binarized event columns based on columns described in a dataset config.
    """
    outcome_data = dataframe_with_outcome.copy()

    # Set up the outcome columns
    outcome_labels = dataset_config['CLINICAL']['OUTCOME_VARIABLES']
    outcome_data = eventOutcomeColumnSetup(dataframe_with_outcome=outcome_data,
                                            outcome_column_label=outcome_labels['event_label'],
                                            standard_column_label=standard_event_label,
                                            event_column_value_mapping=outcome_labels['event_value_mapping']
                                            )
    outcome_data = timeOutcomeColumnSetup(dataframe_with_outcome=outcome_data,
                                           outcome_column_label=outcome_labels['time_label'],
                                           standard_column_label=standard_time_label,
                                           convert_to_years=outcome_labels['convert_to_years']
                                           )
    
    outcome_data = outcome_data[[standard_event_label, standard_time_label]]

    return outcome_data



def predict_with_one_image_type(dataset_config,
                                outcome_data,
                                image_type,
                                signature_name):
    
    full_dataset_name = f"{dataset_config['DATA_SOURCE']}_{dataset_config['DATASET_NAME']}"
    
    # load features
    extraction = dataset_config['EXTRACTION']
    feature_path = dirs.RESULTS / full_dataset_name / "features" / extraction['METHOD'] / Path(extraction['CONFIG']).stem / f"{image_type}_features.csv"
    features = pd.read_csv(feature_path)

    # load signature
    signature = load_signature_config(Path(f"{signature_name}.yaml"))
    return


@click.command()
@click.option('--dataset', type=click.STRING, required=True, help='Name of the dataset to perform prediction with.')
@click.option('--features', type=click.STRING, required=True, help='Feature type to load for prediction.')
@click.option('--signature', type=click.STRING, required=True, help='Name of the signature to perform prediction with. Must have file in config/signatures')
def predict_with_signature(dataset: str,
                           features: str,
                           signature: str):
    
    # Input checking
    if dataset is None:
        message = "Dataset name must be provided."
        logger.error(message)
        raise ValueError(message)
    if features is None:
        message = "Feature type must be provided."
        logger.error(message)
        raise ValueError(message)
    # Input checking
    if signature is None:
        message = "Signature name must be provided."
        logger.error(message)
        raise ValueError(message)


    # get path to dataset config directory
    config_dir_path = dirs.CONFIG / 'datasets'
    
    # Load in dataset configuration settings from provided dataset name
    dataset_config = loadImageDatasetConfig(dataset, config_dir_path)
    dataset_name = dataset_config['DATASET_NAME']
    full_dataset_name = f"{dataset_config['DATA_SOURCE']}_{dataset_config['DATASET_NAME']}"

    logger.info(f"Performing prediction with {signature} signature on {dataset_name}.")

    # load clinical metadata
    clinical_data = clinical_data_setup(dataset_config, full_dataset_name)

    if dataset_config['ANALYSIS']['TRAIN_TEST_SPLIT']['split']:
        split_data = prediction_data_splitting(dataset_config, clinical_data)

    outcome_data = outcome_data_setup(dataset_config, clinical_data)

    return predict_with_one_image_type(dataset_config, outcome_data= outcome_data, image_type='original_full', signature_name=signature)

if __name__ == "__main__":
    predict_with_signature()
