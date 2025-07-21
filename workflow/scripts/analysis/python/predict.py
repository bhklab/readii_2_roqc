from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from damply import dirs

from readii.utils import logger

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





if __name__ == "__main__":
    print(load_signature_config("lasso_10_NSCLC-Radiomics.yaml"))