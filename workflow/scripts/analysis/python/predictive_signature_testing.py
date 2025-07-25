from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from readii.io.loaders import loadFileToDataFrame, loadImageDatasetConfig
from readii.process.label import (
    eventOutcomeColumnSetup,
    getPatientIdentifierLabel,
    timeOutcomeColumnSetup,
)
from readii.process.split import splitDataByColumnValue
from readii.process.subset import getPatientIntersectionDataframes
from sksurv.metrics import concordance_index_censored

# ### Set up variables from config file for the dataset
# Load in the configuration file
config = loadImageDatasetConfig("HEAD-NECK-RADIOMICS-HN1", Path("../../../config/datasets"))

# Initialize dataset parameters
CLINICAL_DATA_FILE = config["CLINICAL_FILE"]
OUTCOME_VARIABLES = config["OUTCOME_VARIABLES"]

DATASET_NAME = config["DATA_SOURCE"] + "_" + config["DATASET_NAME"]

RANDOM_SEED = 10

# general data directory path setup
DATA_DIR_PATH = Path("../../../data")
RAW_DATA_PATH = DATA_DIR_PATH / "rawdata" / DATASET_NAME
PROC_DATA_PATH = DATA_DIR_PATH / "procdata" / DATASET_NAME
RESULTS_DATA_PATH = DATA_DIR_PATH / "results" / DATASET_NAME

# train test split settings
TRAIN_TEST_SPLIT = config["TRAIN_TEST_SPLIT"]

pyradiomics_settings = "pyradiomics_original_plus_aerts"

signature_name = "lasso_10_signature"
signature_dir = PROC_DATA_PATH / "signature_data"
signature_dir.mkdir(parents=True, exist_ok=True)

signature_results_dir = RESULTS_DATA_PATH / "signature_performance"
signature_results_dir.mkdir(parents=True, exist_ok=True)


# # Existing signature loading and processing
# ## Load signature from yaml file
# Path to the yaml file in the rawdata directory
radiomic_signature_yaml = RAW_DATA_PATH.parent / "cph_weights_radiomic_signature" / "aerts_original.yaml"

try:
    with open(radiomic_signature_yaml, 'r') as f:
        yaml_data = yaml.safe_load(f)
        if not isinstance(yaml_data, dict):
            raise TypeError("ROI match YAML must contain a dictionary")
        radiomic_signature = pd.Series(yaml_data['signature'])
except Exception as e:
    print(f"Error loading YAML file: {e}")
    raise



# ## Clinical data loading and processing
# Load clinical data file
clinical_data = loadFileToDataFrame((RAW_DATA_PATH / "clinical" / CLINICAL_DATA_FILE))
clinical_pat_id = getPatientIdentifierLabel(clinical_data)

if TRAIN_TEST_SPLIT['split']:
    split_data = splitDataByColumnValue(clinical_data,
                                        split_col_data=TRAIN_TEST_SPLIT['split_variable'],
                                        impute_value=TRAIN_TEST_SPLIT['impute'])
    # TODO: don't default to test string
    clinical_data = split_data['test']

# Load the Med-ImageTools index to use for mapping TCIA IDs to local file names
mit_index = loadFileToDataFrame((RAW_DATA_PATH / "images" / config["MIT_INDEX_FILE"]))

# Set up SampleID if it doesn't exist in the med-imagetools index file
if 'SampleID' not in mit_index.columns:
    mit_index['SampleID'] = mit_index["PatientID"].astype(str) + "_" + mit_index['SampleNumber'].astype(str).str.zfill(4)

# SampleID is local file name
# PatientID is TCIA ID
id_map = mit_index['SampleID']
id_map.index = mit_index["PatientID"]
id_map = id_map.drop_duplicates()

# Map the SampleIDs to the clinical data and add as a column for intersection
clinical_data['SampleID'] = clinical_data[clinical_pat_id].map(id_map)
clinical_data = clinical_data.set_index('SampleID')

### Set up outcome columns in clinical data
clinical_data = eventOutcomeColumnSetup(dataframe_with_outcome=clinical_data,
                                        outcome_column_label=OUTCOME_VARIABLES["event_label"],
                                        standard_column_label="survival_event_binary",
                                        event_column_value_mapping=OUTCOME_VARIABLES["event_value_mapping"],
                                        )
clinical_data = timeOutcomeColumnSetup(dataframe_with_outcome=clinical_data,
                                       outcome_column_label=OUTCOME_VARIABLES["time_label"],
                                       standard_column_label="survival_time_years",
                                       convert_to_years=OUTCOME_VARIABLES["convert_to_years"])


performance_results = list()
image_types = {str(csv_path.name).removesuffix("_features.csv") for csv_path in sorted(RESULTS_DATA_PATH.rglob("**/*_features.csv"))}

for image_type in image_types:
    print(f"Processing image type: {image_type}")
    # ## Feature data loading and processing
    raw_feature_data = loadFileToDataFrame((RESULTS_DATA_PATH / pyradiomics_settings / f"{image_type}_features.csv"))

    raw_feature_data = raw_feature_data.rename(columns={"ID": "SampleID"})
    # Set the index to SampleID
    raw_feature_data = raw_feature_data.set_index('SampleID')

    # ## Intersect clinical and feature data
    clinical_data_subset, pyrad_subset = getPatientIntersectionDataframes(clinical_data, raw_feature_data, need_pat_index_A=False, need_pat_index_B=False)

    # ## Get just features in radiomic signature
    signature_feature_data = pyrad_subset[radiomic_signature.index]

    # # Prediction Modelling with existing signature weights
    feature_hazards = signature_feature_data.dot(radiomic_signature)

    # Calculate the concordance index
    cindex, _concordant, _discordant, _tied_risk, _tied_time = concordance_index_censored(
        event_indicator = clinical_data_subset['survival_event_binary'].astype(bool),
        event_time = clinical_data_subset['survival_time_years'],
        estimate = feature_hazards,
        )


    # ### Bootstrap to get confidence intervals
    sampled_cindex_bootstrap = []

    hazards_and_outcomes = pd.DataFrame({
        'hazards': feature_hazards,
        'survival_event_binary': clinical_data_subset['survival_event_binary'],
        'survival_time_years': clinical_data_subset['survival_time_years']
    }, index=clinical_data_subset.index)

    bootstrap_count = 1000

    for _idx in range(bootstrap_count):
        sampled_results = hazards_and_outcomes.sample(n=hazards_and_outcomes.shape[0], replace=True)

        sampled_cindex, _concordant, _discordant, _tied_risk, _tied_time = concordance_index_censored(
            event_indicator = sampled_results['survival_event_binary'].astype(bool),
            event_time = sampled_results['survival_time_years'],
            estimate = sampled_results['hazards'],
            )
        
        sampled_cindex_bootstrap.append(sampled_cindex)

    lower_confidence_interval = np.percentile(sampled_cindex_bootstrap, 2.5)  
    upper_confidence_interval = np.percentile(sampled_cindex_bootstrap, 97.5)  

    performance_results = performance_results + [[DATASET_NAME, image_type, cindex, lower_confidence_interval, upper_confidence_interval]]

performance_df = pd.DataFrame(performance_results, columns=["Dataset", "Image_Type", "C-index", "Lower CI", "Upper CI"])
performance_df = performance_df.sort_values(by=["Image_Type"])
performance_df.to_csv(RESULTS_DATA_PATH / "signature_performance" / f"{signature_name}.csv", index=False)


