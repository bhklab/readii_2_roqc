
import logging
from pathlib import Path

import pandas as pd
from damply import dirs

# from readii.utils import logger
from readii.io.loaders import loadFileToDataFrame
from readii.process.label import (
    eventOutcomeColumnSetup,
    timeOutcomeColumnSetup,
)
from readii.process.split import splitDataByColumnValue
from readii.process.subset import getPatientIntersectionDataframes, selectByColumnValue

from readii_2_roqc.utils.loaders import load_signature_config
from readii_2_roqc.utils.metadata import insert_r2r_index

logger = logging.getLogger(__name__)


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
        match split.upper():
            case 'TRAIN':
                keep_data = {split_info['split_variable']: [split_info['train_label']]}
            case 'TEST':
                keep_data = {split_info['split_variable']: [split_info['test_label']]}
            case _:
                message = f"Invalid split={split!r}. Expected 'TRAIN' or 'TEST'."
                logger.error(message)
                raise ValueError(message)
        
        clinical_data = selectByColumnValue(clinical_data,
                                            include_col_values=keep_data)

    return clinical_data



def outcome_data_setup(dataset_config: dict,
                       dataframe_with_outcome: pd.DataFrame,
                       standard_event_label : str | None = None,
                       standard_time_label : str | None = None
                       ) -> pd.DataFrame:
    """Set up time and event prediction label columns based on settings described in a dataset config.
    """
    outcome_data = dataframe_with_outcome.copy()
    
    # Set up the outcome columns
    outcome_labels = dataset_config['CLINICAL']['OUTCOME_VARIABLES']

    # binary event label setup
    # can be used for binary or survival prediction labelling
    event_label = outcome_labels['event_label']
    
    standard_labels = []

    if event_label is None:
        logger.debug('No outcome event label passed for setup. Using time label only.')        
        standard_event_label = None
    
    else:
        mapping = outcome_labels.get('event_value_mapping')
        if not isinstance(mapping, dict):
            message = ("CLINICAL.OUTCOME_VARIABLES.event_value_mapping in config must be a dict.")
            logger.error(message)
            raise ValueError(message)

        # Handling if the event labels are a string
        if pd.api.types.is_object_dtype(outcome_data[event_label]):
            mapped = outcome_data[event_label].map(mapping)

            if mapped.isna().any():  
                bad_vals = outcome_data.loc[mapped.isna(), outcome_labels['event_label']].unique()[:10]  
                message = f"Unmapped event values: {bad_vals}. Update event_value_mapping."  
                logger.error(message)  
                raise ValueError(message)
            
            outcome_data[standard_event_label] = mapped.astype(int) 
        else: 
            # handling if the event labels are any other type 
            outcome_data = eventOutcomeColumnSetup(dataframe_with_outcome=outcome_data,
                                                outcome_column_label=event_label,
                                                standard_column_label=standard_event_label,
                                                event_column_value_mapping=mapping
                                                )
        # add the standardized event label to the list of labels to select at the end of the function
        standard_labels.append(standard_event_label)

    # continuous time label setup      
    # can be used for regression or survival prediction labelling
    time_label = outcome_labels['time_label']
    if time_label is None:
        if event_label is None:
            message = "No outcome labels set in this config. Cannot set up data for prediction labelling."
            logger.error(message)
            raise ValueError(message)
        else:
            logger.debug('No outcome time label passed for setup. Using event label only.')
            standard_time_label = None
    
    else:
        # Convert the time column to years if specified in config and rename to standard name
        outcome_data = timeOutcomeColumnSetup(dataframe_with_outcome=outcome_data,
                                              outcome_column_label=time_label,
                                              standard_column_label=standard_time_label,
                                              convert_to_years=outcome_labels['convert_to_years']
                                              )
        
        # add the standardized time label to the list of labels to select at the end of the function
        standard_labels.append(standard_time_label)
    
    # Select out the standardized outcome columns
    select_outcome_data = outcome_data[standard_labels]

    return select_outcome_data


def prediction_data_setup(dataset_config : dict,
                          feature_file : Path,
                          signature_name : str | None = None,
                          split : str | None = None,
                          standard_event_label : str | None = None,
                          standard_time_label : str | None = None
                          ):
    """Set up the feature and label data for prediction
       Clinical data is loaded, SampleID column added as index, subsetted by inclusion/exclusion variables and into the train or test cohort defined in the dataset config.
       Outcome data (time to event and event status) is extracted from the clinical data; event is converted to numeric, time is converted to years if needed.
       Feature data is loaded and intersected with outcome data to get overlap between the two.
       If a signature is provided, feature data is subsetted to just those features.
    """
    # load clinical metadata
    clinical_data = clinical_data_setup(dataset_config, split = split)

    # get outcome variable data from clinical data
    outcome_data = outcome_data_setup(dataset_config,
                                      clinical_data,
                                      standard_event_label=standard_event_label,
                                      standard_time_label=standard_time_label)

    # Load feature data to create signature with
    feature_data = loadFileToDataFrame(feature_file)

    # Set index in feature data to match outcome data
    feature_data = feature_data.set_index(['SampleID'])
    
    # Intersect outcome and feature data to get overlapping SampleIDs
    outcome_subset, feature_subset = getPatientIntersectionDataframes(outcome_data,
                                                                      feature_data,
                                                                      need_pat_index_A=False,
                                                                      need_pat_index_B=False)

    if signature_name is not None:
        # Load signature as a pd.Series with the index being the feature names
        signature = load_signature_config(signature_name)
        # Select out the features specified in the signature
        feature_subset = get_signature_features(feature_subset, signature)

    return feature_subset, outcome_subset



def prediction_data_splitting(dataset_config: dict,
                              data : pd.DataFrame,
                              ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
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
        logger.debug('Split setting is set to False. Returning (data, None)')
        return data, None
    


def get_signature_features(feature_data : pd.DataFrame,
                           signature : pd.Series
                           ) -> pd.DataFrame:
    """Get just the feature values for the features listed in the signature"""
    # Check that signature features are in the feature_data provided
    missing = [f for f in signature.index if f not in feature_data.columns]  
    if missing:  
        preview = ", ".join(missing[:10]) + ("..." if len(missing) > 10 else "")  
        message = f"{len(missing)} signature feature(s) not found in feature_data: {preview}"  
        logger.error(message)  
        raise KeyError(message)  
    
    return feature_data[signature.index]