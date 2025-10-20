import click
import logging
import numpy as np
import yaml

from damply import dirs
from pathlib import Path
from readii.utils import logger
from readii_2_roqc.utils.loaders import load_dataset_config
from readii_2_roqc.utils.analysis import prediction_data_setup
from sksurv.linear_model import CoxPHSurvivalAnalysis

def save_signature(dataset_name:str,
                   signature_name:str,
                   signature_coefficients:dict[str, np.float64],
                   overwrite:bool = False):

    # add dataset name to front of signature incase same signature features are used on another dataset
    save_signature_name = dataset_name + "_" + signature_name
    
    # add or change file suffix to yaml if not already there
    if Path(save_signature_name).suffix not in {".yaml", ".yml"}:
        save_signature_name = Path(save_signature_name).with_suffix(".yaml").name

    # setup full output path with 
    save_signature_path = dirs.CONFIG / "signatures" / save_signature_name

    if save_signature_path.exists() and not overwrite:
        logger.info(f"Signature file already exists at {save_signature_path}. Set overwrite to True if you wish to update it.")
        return save_signature_path
    
    else:
        try:
            # create folder structure if it doesn't exist
            save_signature_path.parent.mkdir(parents=True, exist_ok=True)
            signature_formatted = {'signature': signature_coefficients}

            # write out the signature with coefficients
            with open(save_signature_path, 'w', encoding='utf-8') as outfile:
                yaml.safe_dump(signature_formatted, outfile, default_flow_style=False)
        except Exception:
            message = f"Error occurred saving the {save_signature_name} signature."
            logger.exception(message)
            raise

        return save_signature_path


# def save_predictions(dataset_name:str,
#                      model)


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
    coefficients : dict[str, np.float64]
        Weights for each of the features used to fit the CPH
    hazards : list
        Hazard value for each sample
    cidx : float
        Harrell's concordance index for the predictions of the set used to fit the model
    """
    # Validate required columns exist
    required_cols = {'survival_event_binary', 'survival_time_years'}
    missing = required_cols.difference(outcome_data.columns)  
    if missing:
        message = f"Outcome data missing required columns: {sorted(missing)}"
        logger.error(message)
        raise KeyError(message)

    # Convert outcome labels to a structured array
    outcome_arr = np.array(
        outcome_data[['survival_event_binary', 'survival_time_years']].to_records(
            index=False, 
            column_dtypes={'survival_event_binary': 'bool', 'survival_time_years': 'float'}
        )
    )
    
    # Convert feature data to array for fitting and evaluation
    feature_arr = feature_data.to_numpy()

    estimator = CoxPHSurvivalAnalysis().fit(feature_arr, outcome_arr)

    coefficients = {
        feature_name:value.item() 
        for (feature_name, value) in zip(feature_data.columns, estimator.coef_, strict=True)}
    
    hazards = estimator.predict(feature_arr)
    cidx = estimator.score(feature_arr, outcome_arr)
    # Return fitted signature feature coefficients, hazards, and c-index for the fitting data
    return coefficients, hazards, cidx



@click.command()
@click.argument('dataset', type=click.STRING)
@click.argument('features', type=click.STRING)
@click.argument('model', type=click.Choice(['cph']))
@click.option('--signature', type=click.STRING, default=None)
@click.option('--image_type', type=click.STRING, default="original_full")
@click.option('--split', type=click.STRING, default=None)
@click.option('--overwrite', is_flag=True, default=False, help="Overwrite existing outputs if present.")
def fit_model(dataset:str,
              features:str,
              model:str,
              signature:str | None = None,
              image_type:str = 'original_full',
              split:str | None = None,
              overwrite:bool = False):
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
    split : str | None (default = None)
        Train or test split to use for fitting the model. Must be TRAIN, TEST, or None.
    overwrite : bool (defaul = False)
        Used to determine if outputs should be overwritten if file already exists.
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
        # Check that signature file exists (.yaml or .yml)  
        sig_dir = dirs.CONFIG / "signatures"  
        sig_yaml = sig_dir / f"{signature}.yaml"  
        sig_yml = sig_dir / f"{signature}.yml"  
        if not (sig_yaml.exists() or sig_yml.exists()):  
            message = f"Signature file '{signature}.yaml|.yml' not found in {sig_dir}." 
            logger.error(message)
            raise FileNotFoundError(message)
    if split == 'None':
        split = None

    # Load in dataset configuration settings from provided dataset name
    dataset_config, dataset_name, full_data_name = load_dataset_config(dataset)
    logger.info(f"Loaded {dataset_name} for model fitting.")

    logger.info("Listing image types ")
    # Get image types from results of feature extraction
    # two **/** in the pattern cover the feature type and image type processed
    image_type_feature_file_list = sorted(Path(dirs.RESULTS / full_data_name / "features").rglob(f"**/**/{features}/{image_type}_features.csv"))

    if len(image_type_feature_file_list) == 0:  
        message = f"No feature file found for '{features}' from '{image_type}' under {dirs.RESULTS / full_data_name / 'features'}."  
        logger.error(message)  
        raise FileNotFoundError(message) 
    if len(image_type_feature_file_list) > 1:
        message = f"Multiple feature files found for {features} from {image_type}. Can only have one to process."
        logger.error(message)
        raise ValueError(message)
    
    feature_file = image_type_feature_file_list[0]

    logger.info(f"Setting up data for prediction.")
    feature_data, outcome_data = prediction_data_setup(dataset_config,
                                                       feature_file,
                                                       signature,
                                                       split)
    
    match model:
        case 'cph':
            coefficients, predictions, cidx = fit_cph(feature_data, outcome_data)
            logger.info("Fitted CPH: %d samples, c-index=%.4f", len(predictions), cidx)
        case '_':
            message = f"{model} type has not been implemented yet. Try another model please."
            logger.error(message)
            raise NotImplementedError(message)

    if signature is None:
        signature_name = "all_features"
    else:
        signature_name = signature
        
    save_signature(dataset_name, 
                   signature_name = signature_name, 
                   signature_coefficients=coefficients, 
                   overwrite = overwrite)


    return None






if __name__ == "__main__":
    fit_model()