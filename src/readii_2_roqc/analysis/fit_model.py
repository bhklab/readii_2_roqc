import click
import logging
import numpy as np
import pandas as pd

from damply import dirs
from pathlib import Path
from readii.io.loaders import loadFileToDataFrame
from readii.process.subset import getPatientIntersectionDataframes
from readii_2_roqc.utils.loaders import load_signature_config, load_dataset_config
from readii_2_roqc.analysis.predict import clinical_data_setup, outcome_data_setup
from sksurv.linear_model import CoxPHSurvivalAnalysis

def get_signature_features(feature_data : pd.DataFrame,
                           signature : pd.DataFrame
                           ) -> pd.DataFrame:
    """Get just the feature values for the features listed in the signature"""

    return feature_data[signature.index]


def fit_cph(feature_data,
            outcome_data):
    """Fit a CoxPH Survival Analysis model and return predicted risks.
    
    Parameters
    ----------
    feature_data : array-like, shape = (n_samples, n_features)
        Data matrix of feature values to fit the model on.
    event_time_label : pd.DataFrame
        Labels for feature data with first column as survival event and second column as survival time
    Returns
    -------
    """
    #TODO: add check that the survival labels actually exist in the dataframe

    # Convert outcome labels to a structured array
    outcome_arr = np.array(outcome_data.to_records(index=False, 
                                          column_dtypes={'survival_event_binary': 'bool', 
                                                         'survival_event_time': 'float'}))

    estimator = CoxPHSurvivalAnalysis().fit(feature_data.to_numpy(), outcome_arr)

    coefficients = dict(zip(feature_data.columns, estimator.coef_))
    hazards = estimator.predict(feature_data)
    cidx = estimator.score(feature_data, outcome_arr)

    # TODO: Save out signature, hazards, and c-index

    # Return fitted signature feature coefficients, hazards, and c-index for the fitting data
    return coefficients, hazards, cidx


def prediction_data_setup(dataset_config : dict,
                          feature_file : Path,
                          signature_name : str | None):
    """Set up the feature and label data for prediction"""
    # load clinical metadata
    clinical_data = clinical_data_setup(dataset_config)
    # get outcome variable data from clinical data
    outcome_data = outcome_data_setup(dataset_config, clinical_data)

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



@click.command()
@click.argument('dataset', type=click.STRING)
@click.argument('features', type=click.STRING)
@click.argument('model', type=click.Choice(['cph']))
@click.option('--signature', type=click.STRING, default=None)
@click.option('--image_type', type=click.STRING, default="original_full")
def fit_model(dataset:str,
              features:str,
              model:str,
              signature:str | None = None,
              image_type:str = 'original'):
    """Fit a specified model with a signature list of features.

    Parameters
    ----------
    dataset : str
        Name of the dataset to perform prediction with.
    features : str
        Feature type to load for prediction. Will match a feature extraction settings file in config.
    model : str
        Type of model to fit. Options: 
            * 'cph': Fit a scikit-survival CoxPHSurvivalAnalysis model
    signature : str (default = None)
        Name of signature with list of features to use for model fitting. Must exist in config/signatures.
        If None is passed, will use all of the features in the model.
    image_type : str (default = "original_full")
        Image type to use for model fitting. Defaults to the original features, but can be set with any of the
        negative control options from READII.        
    """
    logger = logging.getLogger(__name__)  
    dirs.LOGS.mkdir(parents=True, exist_ok=True)  
    logging.basicConfig(  
        filename=str(dirs.LOGS / f"{dataset}_fit_{model}_{signature}.log"),  
        encoding='utf-8',  
        level=logging.DEBUG,  
        force=True  
    )

    # Input checking
    if dataset is None:
        message = "Dataset name must be provided."
        logger.error(message)
        raise ValueError(message)
    if features is None:
        message = "Feature type must be provided."
        logger.error(message)
        raise ValueError(message)
    if model is None:
        message = "Model type must be provided."
        logger.error(message)
        raise ValueError(message)
    if signature is not None:
        # Check that signature file exists
        if not Path(dirs.CONFIG / "signatures" / f"{signature}.yaml").exists():
            message = f"{signature}.yaml file does not exist in the config/signatures directory."
            logger.error(message)
            raise FileNotFoundError(message)

    # Load in dataset configuration settings from provided dataset name
    dataset_config, dataset_name, full_data_name = load_dataset_config(dataset)
    logger.info(f"Loaded {dataset_name} for model fitting.")

    logger.info("Listing image types ")
    # Get image types from results of feature extraction
    # two **/** in the pattern cover the feature type and image type processed
    image_type_feature_file_list = sorted(Path(dirs.RESULTS / full_data_name / "features").rglob(pattern = f"**/**/{features}/{image_type}_features.csv"))

    if len(image_type_feature_file_list) > 1:
        message = f"Multiple feature files found for {features} from {image_type}. Can only have one to process."
        logger.error(message)
        raise ValueError(message)
    
    feature_file = image_type_feature_file_list[0]

    logger.info(f"Setting up data for prediction.")
    feature_data, outcome_data = prediction_data_setup(dataset_config,
                                                           feature_file,
                                                           signature)
    
    match model:
        case 'cph':
            coefficients, hazards, cidx = fit_cph(feature_data, outcome_data)
            
            print(coefficients)
            print(hazards)
            print(cidx)
        case '_':
            message = f"{model} type has not been implemented yet. Try another model please."
            logger.error(message)
            raise NotImplementedError(message)

    return None
    



if __name__ == "__main__":
    fit_model()