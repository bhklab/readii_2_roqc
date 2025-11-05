import logging
from pathlib import Path

# from readii.utils import logger
import numpy as np

logger = logging.getLogger(__name__)

REGIONS = {'full', 'roi', 'non_roi'}
PERMUTATIONS = {'shuffled', 'sampled', 'randomized'}
CROP = {'cube', 'bbox', 'centroid'}

def get_resize_string(resize: list | tuple | np.ndarray | int) -> str:
    """Convert resize numeric argument into readable string format"""
    if isinstance(resize, (list, tuple, np.ndarray)):
        return '_'.join(str(val) for val in resize)
    elif isinstance(resize, int):
        return f"{resize}_{resize}_{resize}"
    else:
        message = f"Improper resize input type. Must be a list, tuple, array, or int. Current type: {type(resize)}"
        logger.error(message)
        raise TypeError(message)


def get_readii_index_filepath(dataset_config:dict,
                              readii_image_dir:Path):
    """Construct the full filepath to the READII image index that lists all the processed images from running make_negative_controls.
       This function requires that the READII index file exist to find the filepath for uncropped images.

    Parameters
    ----------
    dataset_config:dict
        Settings dictionary for the dataset being processed from running `load_dataset_config`
    readii_image_dir:Path
        Path to the READII outputs. Would be the same as what was passed as the output_dir to `image_preprocessor` or used to construct the NIFTIWriter for `negative_control_generator`.
    
    Returns
    -------
    readii_index_path:Path
        Path to the READII index file
    """
    # Get dataset name from config settings
    dataset_name = dataset_config['DATASET_NAME']

    # Load the requested image processing settings from configuration
    _regions, _permutations, crop, resize = get_readii_settings(dataset_config)

    try:
        # Path to find existing readii index output for checking existing outputs
        if crop is not None and resize is not None:
            readii_index_filepath = readii_image_dir.glob(f"{crop}_{get_resize_string(resize)}/readii_{dataset_name}_index.csv").__next__()
        else:
            readii_index_filepath = readii_image_dir.glob(f"original_*/readii_{dataset_name}_index.csv").__next__()
    except StopIteration:
        message = "No READII index file was found for the specified settings"
        logger.warning(message)
        raise FileNotFoundError(message) from None

    return readii_index_filepath


def get_extraction_index_filepath(dataset_config:dict,
                                  extract_features_dir:Path):
    # Get dataset name from config settings
    dataset_name = dataset_config['DATASET_NAME']

    extract_method = extract_features_dir.stem

    # Load the requested image processing settings from configuration
    _regions, _permutations, crop, resize = get_readii_settings(dataset_config)

    try:
        # Path to find existing readii index output for checking existing outputs
        if crop is not None and resize is not None:
            extract_index_filepath = extract_features_dir.glob(f"{crop}_{get_resize_string(resize)}/{extract_method}_{dataset_name}_index.csv").__next__()
        else:
            extract_index_filepath = extract_features_dir.glob(f"original_*/{extract_method}_{dataset_name}_index.csv").__next__()
    except StopIteration:
        message = f"No {extract_method} index file was found for the specified settings"
        logger.warning(message)
        raise FileNotFoundError(message) from None

    return extract_index_filepath


def check_setting_superset(setting_list: set, 
                           setting_request: set | list | None
                           ) -> set | list | None:
    """
    Check if the requested settings are all in the global settings list for READII.

    Parameters
    ----------
    setting_list: set
        Set of allowed values for a READII setting. Should be set using the global sets in this file.
    setting_request: set | list | None
        Set of requested values for a READII setting.

    Returns
    -------
    Returns the settings requested if they are a subset of the implemented options
    Returns None if the setting request is any kind of blank ([], "", None)
    
    Raises
    ------
    ValueError if the requested settings are not a subset of the allowed settings.
    """
    if setting_request == [] or setting_request is None or setting_request == "":
        return None
    if not setting_list.issuperset(setting_request):
        message = f"Requested settings ({setting_request}) is not a subset of the allowed settings ({setting_list})."
        logger.error(message)
        raise ValueError(message)

    else:
        return setting_request



def get_readii_settings(dataset_config: dict) -> tuple[list, list, str, int | list[int]]:
    """Extract READII settings from a configuration dictionary.
    
    Parameters
    ----------
    dataset_config : dict
        Configuration dictionary read in with `loadImageDatasetConfig` containing READII settings
    
    Returns
    -------
    tuple  
        Returns a tuple of four elements:  
        - regions: list[str] of regions to process  
        - permutations: list[str] of permutations to apply  
        - crop: str indicating the selected crop ("" when unset)  
        - resize: int for isotropic resize, or list[int] of length 0 or 3 
    """
    readii_config = dataset_config['READII']
    if 'IMAGE_TYPES' not in readii_config:
        message = "READII configuration must contain 'IMAGE_TYPES'."
        logger.error(message)
        raise KeyError(message)
    
    regions = readii_config['IMAGE_TYPES']['regions']
    # Confirm requested regions are available settings
    regions = check_setting_superset(REGIONS, regions)
        
    permutations = readii_config['IMAGE_TYPES']['permutations']
    # Confirm requested permutations are available settings
    permutations = check_setting_superset(PERMUTATIONS, permutations)

    crop = readii_config['IMAGE_TYPES']['crop']
    if crop is not None and crop != []:
        # Confirm requested crop is an available setting
        crop = check_setting_superset(CROP, crop)
        # Get single crop value out of list format
        crop = crop[0]
    else:
        crop = None

    resize = readii_config['IMAGE_TYPES']['resize']
    if resize is None:
        resize = None
    elif isinstance(resize, list):
        match len(resize):
            case 1: 
                resize = resize[0]
            case 3: 
                pass # resize already has 3 elements
            case 0: 
                pass # resize is already empty
            case _: 
                message = f"READII resize must be a single int, or list of three ints (e.g. [50, 50, 50]). Current value: {resize}"
                logger.error(message)
                raise TypeError(message)
    elif not isinstance(resize, int):
        message = f"READII resize must be a single int, or list of three values (e.g. [50, 50, 50]). Current value: {resize}"
        logger.error(message)
        raise TypeError(message)

    return regions, permutations, crop, resize