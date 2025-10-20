import numpy as np
import pandas as pd
import yaml

from damply import dirs
from pathlib import Path
from readii.utils import logger



def save_signature(dataset_name:str,
                   signature_name:str,
                   signature_coefficients:dict[str, np.float64],
                   model_type:str = None,
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