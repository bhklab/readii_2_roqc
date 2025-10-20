from damply import dirs
from pathlib import Path
from readii.io.loaders import loadImageDatasetConfig
from readii.process.config import get_full_data_name
from readii.utils import logger
from readii.image_processing import alignImages, flattenImage

import pandas as pd
import SimpleITK as sitk
import yaml


def load_dataset_config(dataset:str):
    if dataset is None:
            message = "Dataset name must be provided."
            logger.error(message)
            raise ValueError(message)

    # get path to dataset config directory
    config_dir_path = dirs.CONFIG / 'datasets'
    
    # Load in dataset configuration settings from provided dataset name
    dataset_config = loadImageDatasetConfig(dataset, config_dir_path)

    dataset_name = dataset_config['DATASET_NAME']
    full_data_name = get_full_data_name(config_dir_path / dataset)

    return dataset_config, dataset_name, full_data_name



def load_image_and_mask(image_path:Path, 
                        mask_path:Path | None = None
                        ) -> sitk.Image | tuple[sitk.Image, sitk.Image]:
    """ Load in an image (and optionally a mask) from nifti files and perform flatten and alignment preprocessing.
        This removes any dimensions with size 1, and aligns the origin, direction, and spacing of the mask to the image.
    
    Parameters
    ----------
    image_path:Path
        Path to the image NIfTI file (.nii.gz) to load.
    mask_path:Path or None, optional
        Path to the mask NIfTI file (.nii.gz). If provided, mask is loaded and aligned to the image.  

    Returns
    -------
    sitk.Image | tuple[sitk.Image, sitk.Image]  
        If mask_path is None: returns the aligned image  
        If mask_path is provided: returns tuple of (image, mask) 
    
    Notes  
    -----  
    The function applies alignImages after flattenImage to maintain proper  
    origin, spacing, and direction metadata.  
    """  
    # Load in image
    raw_image = sitk.ReadImage(image_path)
    # Remove extra dimension of image, set origin, spacing, direction to original
    image = alignImages(raw_image, flattenImage(raw_image)) 

    if mask_path:
        # Load in mask
        raw_mask = sitk.ReadImage(mask_path)
        mask = alignImages(image, flattenImage(raw_mask))
        return image, mask
    
    return image



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
        If the input file does not end in ".yaml". 
    """
    p = Path(file)  
    if p.suffix == "":  
        file = p.with_suffix(".yaml")  
    elif p.suffix not in {".yaml", ".yml"}:  
        message = f"Signature file must have .yaml or .yml extension. Got {p.suffix!r}."  
        logger.error(message)  
        raise ValueError(message)  

    signature_file_path = dirs.CONFIG / "signatures" / file

    try:
        with signature_file_path.open('r', encoding='uft-8') as f:
            yaml_data = yaml.safe_load(f)
            if not isinstance(yaml_data, dict):
                message = "ROI match YAML must contain a dictionary"
                logger.error(message)
                raise TypeError(message)
            if "signature" not in yaml_data:
                message = "Signature YAML missing required key 'signature'."
                logger.error(message)
                raise KeyError(message)
            if not isinstance(yaml_data["signature"], dict): 
                message = "Signature must be a mapping of {feature: weight}."
                logger.error(message)
                raise TypeError(message)
            
            signature = pd.Series(yaml_data['signature'])

    except Exception:
        message = f"Error loading signature YAML file at {signature_file_path}"
        logger.exception(message)
        raise

    return signature