from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from damply import dirs
import click
from joblib import Parallel, delayed
from readii.utils import logger
from readii.io.loaders import loadImageDatasetConfig, loadFileToDataFrame
from readii.process.config import get_full_data_name
from readii.process.label import (
    eventOutcomeColumnSetup,
    getPatientIdentifierLabel,
    timeOutcomeColumnSetup,
)
from readii.process.split import splitDataByColumnValue
from readii.process.subset import selectByColumnValue, getPatientIntersectionDataframes
from sksurv.metrics import concordance_index_censored

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
                              data : pd.DataFrame,
                              train_variable : str = 'training',
                              test_variable : str = 'test'
                              ) -> tuple[pd.DataFrame]:
    """Split metadata into train and test for model development and validation purposes"""
    split_settings = dataset_config['ANALYSIS']['TRAIN_TEST_SPLIT']
    if split_settings['split']:
        split_data = splitDataByColumnValue(data,
                                            split_col_data=split_settings['split_variable'],
                                            impute_value=split_settings['impute'])
        return split_data[train_variable], split_data[test_variable]
    else:
        logger.debug('Split setting is set to False. Returning original data.')
        return data
    


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
    # Select out the standardized outcome columns
    outcome_data = outcome_data[[standard_event_label, standard_time_label]]

    return outcome_data


def calculate_signature_hazards(feature_data : pd.DataFrame,
                                signature : pd.DataFrame) -> pd.DataFrame:
    """Calculate the feature hazards with Cox Proportional Hazards by multiplying feature values by the signature weights"""
    # Get signature feature values for the dataset
    signature_feature_data = feature_data[signature.index]

    # Calculate and return the feature hazards
    return signature_feature_data.dot(signature)


def evaluate_signature_prediction(hazards_and_outcomes : pd.DataFrame) -> tuple:
    concordance_evals = concordance_index_censored(
            event_indicator = hazards_and_outcomes['survival_event_binary'],
            event_time = hazards_and_outcomes['survival_time_years'],
            estimate = hazards_and_outcomes['hazards']
           )

    # Return just the cindex, the first value in the tuple returned by concordance_index_censored
    return concordance_evals[0]


def predict_with_one_image_type(feature_data,
                                outcome_data : pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame],
                                signature_name : str,
                                bootstrap : int = 0
                                ) -> pd.DataFrame:
    # load signature
    signature = load_signature_config(Path(f"{signature_name}.yaml"))
    
    # Intersect outcome and feature data to get overlapping SampleIDs
    outcome_subset, feature_subset = getPatientIntersectionDataframes(outcome_data,
                                                                      feature_data,
                                                                      need_pat_index_A=False,
                                                                      need_pat_index_B=False)

    feature_hazards = calculate_signature_hazards(feature_subset, signature)

    # Add hazards as a column to a dataframe with the ground truth time and event labels and SampleID index
    # This will make it easier to evaluate the predictions
    hazards_and_outcomes = outcome_subset.copy()
    hazards_and_outcomes['hazards'] = feature_hazards

    cindex = evaluate_signature_prediction(hazards_and_outcomes)
    lower_confidence_interval = np.nan
    upper_confidence_interval = np.nan

    if bootstrap > 0:
        sampled_cindex = Parallel(n_jobs=-1)(
                        delayed(evaluate_signature_prediction)(
                            hazards_and_outcomes = hazards_and_outcomes.sample(n=hazards_and_outcomes.shape[0], replace=True)
                        )
                        for _idx in range(bootstrap)
                        )

        lower_confidence_interval = np.percentile(sampled_cindex, 2.5)  
        upper_confidence_interval = np.percentile(sampled_cindex, 97.5)

    metrics = [cindex, lower_confidence_interval, upper_confidence_interval]

    return metrics, hazards_and_outcomes


@click.command()
@click.option('--dataset', type=click.STRING, required=True, help='Name of the dataset to perform prediction with.')
@click.option('--features', type=click.STRING, required=True, help='Feature type to load for prediction. Will match a feature extraction settings file in config.')
@click.option('--signature', type=click.STRING, required=True, help='Name of the signature to perform prediction with. Must have file in config/signatures')
def predict_with_signature(dataset: str,
                           features: str,
                           signature: str,
                           bootstrap : int = 0,
                           parallel: bool = False):
    
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

    # TODO: figure out what to do with train/test data
    # if dataset_config['ANALYSIS']['TRAIN_TEST_SPLIT']['split']:
    #     logger.info(f"Splitting data into train and test subsets.")
    #     train_clinical, test_clinical = prediction_data_splitting(dataset_config, clinical_data)
    #     train_outcome, test_outcome = outcome_data_setup(dataset_config, train_clinical), outcome_data_setup(dataset_config, test_clinical)
    #     # Put into tuple for passing to prediction functions as a single argument
    #     outcome_data = (train_outcome, test_outcome)
    # else:

    outcome_data = outcome_data_setup(dataset_config, clinical_data)

    # Get image types from results of feature extraction
    image_type_feature_file_list = sorted(Path(dirs.RESULTS / full_dataset_name / "features").rglob(pattern = f"**/{features}/*_features.csv"))

    # Set up analysis outputs
    prediction_out_dir = dirs.RESULTS / full_dataset_name / "prediction" / signature
    hazards_out_dir = prediction_out_dir / f"hazards"
    hazards_out_dir.mkdir(parents=True, exist_ok=True)

    prediction_data = []
    hazard_data = {}

    for feature_file_path in image_type_feature_file_list:
        image_type = feature_file_path.stem.removesuffix('features.csv')
        feature_data = loadFileToDataFrame(feature_file_path)

        prediction_eval, hazards = predict_with_one_image_type(feature_data = feature_data,
                                                               outcome_data = outcome_data,
                                                               signature_name = signature,
                                                               bootstrap = bootstrap)
        
        prediction_data = prediction_data + [[dataset_name, image_type] + prediction_eval]
        hazard_data[image_type] = hazards
        

    prediction_df = pd.DataFrame(prediction_data,
                                 columns=['Dataset',
                                          'Image_Type',
                                          'C-index',
                                          'Lower_CI',
                                          'Upper_CI'])
    prediction_df = prediction_df.sort_values(by=["Image_Type"])
    prediction_df.to_csv(prediction_out_dir / "prediction_metrics.csv")

    [hazard_df.csv(hazards_out_dir / image_type) for image_type, hazard_df in hazard_data]

    return prediction_df, hazard_data

if __name__ == "__main__":
    predict_with_signature()
