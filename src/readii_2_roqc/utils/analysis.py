
import numpy as np
import pandas as pd

from damply import dirs
from readii.utils import logger
from readii.io.loaders import loadFileToDataFrame
from readii.process.label import (
    eventOutcomeColumnSetup,
    timeOutcomeColumnSetup,
)
from readii.process.subset import selectByColumnValue
from readii.process.split import splitDataByColumnValue
from readii_2_roqc.utils.metadata import insert_r2r_index


def clinical_data_setup(dataset_config: dict,
                       full_data_name : str | None = None,
                       split: str | None = None
                       ) -> pd.DataFrame:
    """Process the clinical data to get outcome variables for use in signature prediction"""
    if full_data_name is None:
        full_data_name = f"{dataset_config['DATA_SOURCE']}_{dataset_config['DATASET_NAME']}"

    # load clinical metadata
    clinical = dataset_config['CLINICAL']
    clinical_path = dirs.RAWDATA / full_data_name / "clinical" / clinical['FILE']
    clinical_data = loadFileToDataFrame(clinical_path)

    # insert the MIT index
    clinical_data = insert_r2r_index(dataset_config, clinical_data)

    # Set the MIT SampleIDs as the index for clinical data
    clinical_data = clinical_data.set_index('SampleID')

    # Drop rows based on exclusion variables in config file
    if len(clinical['EXCLUSION_VARIABLES']) != 0 or len(clinical['INCLUSION_VARIABLES']) != 0:
        clinical_data = selectByColumnValue(clinical_data,
                                            exclude_col_values = clinical['EXCLUSION_VARIABLES'],
                                            include_col_values = clinical['INCLUSION_VARIABLES'])

    # Get the train or test sub-cohort based on config file setup
    if split is not None:
        split_info = dataset_config['ANALYSIS']['TRAIN_TEST_SPLIT']
        match split:
            case 'TRAIN':
                keep_data = {split_info['split_variable']: [split_info['train_label']]}
            case 'TEST':
                keep_data = {split_info['split_variable']: [split_info['test_label']]}
        
        clinical_data = selectByColumnValue(clinical_data,
                                            include_col_values=keep_data)

    return clinical_data



def outcome_data_setup(dataset_config: dict,
                       dataframe_with_outcome: pd.DataFrame,
                       standard_event_label : str = "survival_event_binary",
                       standard_time_label : str = "survival_time_years"
                       ) -> pd.DataFrame:
    """Set up survival time in years and binarized event columns based on columns described in a dataset config.
    """
    outcome_data = dataframe_with_outcome.copy()
    
    # Set up the outcome columns
    outcome_labels = dataset_config['CLINICAL']['OUTCOME_VARIABLES']

    event_variable_type = outcome_data[outcome_labels['event_label']].dtype
    if np.issubdtype(event_variable_type, np.object_):
        # TEMP: handle value mapping to integers
        outcome_data[standard_event_label] = outcome_data[outcome_labels['event_label']].map(outcome_labels['event_value_mapping'])
    else:
        outcome_data = eventOutcomeColumnSetup(dataframe_with_outcome=outcome_data,
                                                outcome_column_label=outcome_labels['event_label'],
                                                standard_column_label=standard_event_label,
                                                event_column_value_mapping=None #outcome_labels['event_value_mapping']
                                                )
    
    outcome_data = timeOutcomeColumnSetup(dataframe_with_outcome=outcome_data,
                                           outcome_column_label=outcome_labels['time_label'],
                                           standard_column_label=standard_time_label,
                                           convert_to_years=outcome_labels['convert_to_years']
                                           )
    # Select out the standardized outcome columns
    outcome_data = outcome_data[[standard_event_label, standard_time_label]]

    return outcome_data



def prediction_data_splitting(dataset_config: dict,
                              data : pd.DataFrame,
                              ) -> tuple[pd.DataFrame]:
    """Split metadata into train and test for model development and validation purposes"""
    split_settings = dataset_config['ANALYSIS']['TRAIN_TEST_SPLIT']
    # Construct the dictionary input expected to describe the split column and labels within it
    split_col_dict = {split_settings['split_variable']: [split_settings['train_label'], split_settings['test_label']]}
    # Check that split setting is true before completing split
    if split_settings['split']:
        split_data = splitDataByColumnValue(data,
                                            split_col_data=split_col_dict,
                                            impute_value=split_settings['impute'])
        return split_data[split_settings['train_label']], split_data[split_settings['test_label']]
    else:
        logger.debug('Split setting is set to False. Returning original data.')
        return data