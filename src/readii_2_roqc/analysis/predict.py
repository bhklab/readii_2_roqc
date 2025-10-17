import click
import logging
import numpy as np
import pandas as pd

from damply import dirs
from joblib import Parallel, delayed
from pathlib import Path
from readii.io.loaders import loadFileToDataFrame
from readii.process.subset import getPatientIntersectionDataframes
from readii.utils import logger
from readii_2_roqc.utils.loaders import load_dataset_config, load_signature_config
from readii_2_roqc.utils.analysis import clinical_data_setup, outcome_data_setup, get_signature_features
from sksurv.metrics import concordance_index_censored


def calculate_signature_hazards(feature_data : pd.DataFrame,
                                signature : pd.DataFrame) -> pd.DataFrame:
    """Calculate the feature hazards with Cox Proportional Hazards by multiplying feature values by the signature weights"""
    # Get signature feature values for the dataset
    signature_feature_data = get_signature_features(feature_data, signature)

    # Calculate and return the feature hazards
    return signature_feature_data.dot(signature)



def evaluate_signature_prediction(hazards_and_outcomes : pd.DataFrame) -> tuple:
    """Get the c-index for a set of predictions/hazards."""
    concordance_evals = concordance_index_censored(
            event_indicator = hazards_and_outcomes['survival_event_binary'].astype(bool),
            event_time = hazards_and_outcomes['survival_time_years'],
            estimate = hazards_and_outcomes['hazards']
           )

    # Return just the cindex, the first value in the tuple returned by concordance_index_censored
    return concordance_evals[0]



def bootstrap_c_index(hazards_and_outcomes: pd.DataFrame,
                      bootstrap_count: int = 1000,             
                     ) -> tuple[list[float], float, float]:
    """Generate confidence intervals by bootstrapping a set of prediction/hazard values by sampling with replacement and calculating metrics."""
    if bootstrap_count < 1:
        message = "Bootstrap count must be a positive integer."
        logger.error(message)
        raise ValueError(message)
    
    bootstrap_cidx = []
    # Bootstrap the prediction results to get confidence intervals
    sampled_cindex = Parallel(n_jobs=-1)(
                    delayed(evaluate_signature_prediction)(
                        hazards_and_outcomes = hazards_and_outcomes.sample(n=hazards_and_outcomes.shape[0], replace=True)
                    )
                    for _idx in range(bootstrap_count)
                    )
    bootstrap_cidx += sampled_cindex
    lower_confidence_interval = np.percentile(sampled_cindex, 2.5)  
    upper_confidence_interval = np.percentile(sampled_cindex, 97.5)
    
    return bootstrap_cidx, lower_confidence_interval, upper_confidence_interval



def predict_with_one_image_type(feature_data: pd.DataFrame,
                                outcome_data : pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame],
                                signature_name : str,
                                bootstrap : int = 0
                                ) -> tuple[list[float], list[float], pd.DataFrame]:
    """Evaluate the outcome prediction performance of a provided signature with a feature dataset from one image type. Optional bootstrapping for confidence intervals."""
    # load signature
    signature = load_signature_config(Path(f"{signature_name}.yaml"))
    # Set index in feature data to match outcome data
    feature_data = feature_data.set_index(['SampleID'])
    
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

    # Calculate the c-index for the predicted hazards
    cindex = evaluate_signature_prediction(hazards_and_outcomes)
    # Initialize the confidence intervals for no bootstrap handling
    lower_confidence_interval = np.nan
    upper_confidence_interval = np.nan
    bootstrap_cidxs = []

    if bootstrap > 0:
        bootstrap_cidxs, lower_confidence_interval, upper_confidence_interval = bootstrap_c_index(hazards_and_outcomes, bootstrap)

    # Combine the metrics into a list
    metrics = [cindex, lower_confidence_interval, upper_confidence_interval]

    return metrics, bootstrap_cidxs, hazards_and_outcomes


DATA_SPLIT_CHOICES = ['TRAIN', 'TEST', 'NONE']

@click.command()
@click.argument('dataset', type=click.STRING)
@click.argument('features', type=click.STRING)
@click.argument('signature', type=click.STRING)
@click.option('--bootstrap', type=click.INT, default=0, help='Number of bootstrap iterations to run for confidence interval generation.')
@click.option('--split', type=click.Choice(DATA_SPLIT_CHOICES), default='NONE', help="Data subset to use for prediction, TRAIN or TEST. Will get settings from dataset config.")
def predict_with_signature(dataset: str,
                           features: str,
                           signature: str,
                           bootstrap: int = 0,
                           split: str = 'NONE'
                           ) ->tuple[pd.DataFrame, dict, dict]:
    """Run outcome prediction of a signature for multiple image types
    
    Parameters
    ----------
    dataset : str
        Name of the dataset to perform prediction with.
    features : str
        Feature type to load for prediction. Will match a feature extraction settings file in config.
    signature : str
        Name of the signature to perform prediction with. Must have file in config/signatures.
    bootstrap : int (default = 0)
        Number of bootstrap iterations to run for confidence interval generation. Default if 0, no bootstrap will be run.
    split : str (default = 'NONE')
        Data subset to use for prediction, TRAIN or TEST. Will get settings from dataset config.

    Returns
    -------
    prediction_df : pd.DataFrame 
    bootstrap_data : dict
    hazard_data : dict
    """
    logger = logging.getLogger(__name__)  
    dirs.LOGS.mkdir(parents=True, exist_ok=True)  
    logging.basicConfig(  
        filename=str(dirs.LOGS / f"{dataset}_predict.log"),  
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
    # Input checking
    if signature is None:
        message = "Signature name must be provided."
        logger.error(message)
        raise ValueError(message)
    # Input checking
    if split == 'NONE':
        split = ""

    # Load in dataset configuration settings from provided dataset name
    dataset_config, dataset_name, full_data_name = load_dataset_config(dataset)
    logger.info(f"Performing prediction with {signature} signature on {dataset_name}.")

    # load clinical metadata
    clinical_data = clinical_data_setup(dataset_config, full_data_name)
    # get outcome variable data from clinical data
    outcome_data = outcome_data_setup(dataset_config, clinical_data)

    # Get image types from results of feature extraction
    # two **/** in the pattern cover the feature type and image type processed
    image_type_feature_file_list = sorted(Path(dirs.RESULTS / full_data_name / "features").rglob(pattern = f"**/**/{features}/*_features.csv"))

    # Set up analysis outputs
    prediction_out_dir = dirs.RESULTS / full_data_name / "prediction" / signature / split

    hazards_out_dir = prediction_out_dir / "hazards"
    hazards_out_dir.mkdir(parents=True, exist_ok=True)

    prediction_data = []
    hazard_data = {}
    bootstrap_data = {}

    for feature_file_path in image_type_feature_file_list:
        image_type = feature_file_path.name.removesuffix('_features.csv')
        logger.info(f"Predicting with {signature} signature for {dataset_name} {image_type} image type.")
        feature_data = loadFileToDataFrame(feature_file_path)

        prediction_metrics, bootstrap_cidx, hazards = predict_with_one_image_type(feature_data = feature_data,
                                                                                outcome_data = outcome_data,
                                                                                signature_name = signature,
                                                                                bootstrap = bootstrap)

        prediction_data = prediction_data + [[dataset_name, image_type] + prediction_metrics]
        hazard_data[image_type] = hazards
        bootstrap_data[image_type] = pd.DataFrame(data = bootstrap_cidx, columns=["C-index"])
        

    prediction_df = pd.DataFrame(prediction_data,
                                 columns=['Dataset',
                                          'Image_Type',
                                          'C-index',
                                          'Lower_CI',
                                          'Upper_CI'])
    prediction_df = prediction_df.sort_values(by=["Image_Type"])
    prediction_df.to_csv(prediction_out_dir / "prediction_metrics.csv", index=False)

    [hazard_df.to_csv(hazards_out_dir / f"{image_type}.csv") for image_type, hazard_df in hazard_data.items()]

    if bootstrap > 0:
        bootstrap_out_dir = prediction_out_dir / f"bootstrap_{bootstrap}"
        bootstrap_out_dir.mkdir(parents=True, exist_ok=True)
        [bootstrap_df.to_csv(bootstrap_out_dir / f"{image_type}.csv") for image_type, bootstrap_df in bootstrap_data.items()]

    return prediction_df, bootstrap_data, hazard_data



if __name__ == "__main__":
    predict_with_signature()