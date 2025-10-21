import numpy as np
import pandas as pd
import yaml

from damply import dirs
from pathlib import Path
from readii.utils import logger

from typing import Any, Mapping


def save_signature(dataset_name:str,
                   signature_name:str,
                   signature_coefficients:Mapping[str, float],
                   model_type:str | None = None,
                   overwrite:bool = False
                   ) -> Path:
    """Save out a radiomic signature used for predictive modelling as a .yaml file
       Will contain the model type (e.g. CoxPHSurvivalAnalysis) and the signature as a dictionary of feature names and weights/coefficients.
    """
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
            signature_formatted = {'model': model_type,
                                   'signature': signature_coefficients}

            # write out the signature with coefficients
            with open(save_signature_path, 'w', encoding='utf-8') as outfile:
                yaml.safe_dump(signature_formatted, outfile, default_flow_style=False)
        except Exception:
            message = f"Error occurred saving the {save_signature_name} signature."
            logger.exception(message)
            raise

        return save_signature_path
    


def save_evaluation(dataset_config:dict[str, Any],
                    evaluation_data:pd.DataFrame,
                    signature:str,
                    split:str | None = None,
                    ) -> tuple[Path, pd.DataFrame]:
    """Save out table of evaluation metrics for a predictive model.
       Return the path to the saved output and the updated evaluation data dataframe with the Dataset and Image Type columns added if they were not present.
    """
    if split is None:
        split = ''
    # Set up analysis outputs
    full_data_name = f"{dataset_config['DATA_SOURCE']}_{dataset_config['DATASET_NAME']}"
    evaluation_out_path = dirs.RESULTS / full_data_name / "prediction" / signature / split / "prediction_metrics.csv"
    evaluation_out_path.parent.mkdir(parents=True, exist_ok=True)

    if "Dataset" not in evaluation_data.columns:
        evaluation_data['Dataset'] = dataset_config['DATASET_NAME']
        
    if "Image_Type" not in evaluation_data.columns:
        evaluation_data["Image_Type"] = 'unknown'
    else:
        evaluation_data = evaluation_data.sort_values(by=["Image_Type"])

    evaluation_data.to_csv(evaluation_out_path, index=False)

    return evaluation_out_path, evaluation_data


def save_predictions(dataset_config:dict[str, Any],
                     prediction_data:dict[str,pd.DataFrame],
                     signature:str,
                     prediction_type:str | None = None,
                     split:str | None = None):
    
    if prediction_type is None:
        prediction_type = 'predictions'
    
    if split is None:
        split = ''

    full_data_name = f"{dataset_config['DATA_SOURCE']}_{dataset_config['DATASET_NAME']}"
    prediction_out_dir = dirs.RESULTS / full_data_name / "prediction" / signature / split / prediction_type
    prediction_out_dir.mkdir(parents=True, exist_ok=True)

    prediction_out_paths = []
    for image_type, prediction_df in prediction_data.items():
        out_path = prediction_out_dir / f"{image_type}.csv"
        prediction_df.to_csv(out_path)
        prediction_out_paths.append(out_path)

    return prediction_out_paths