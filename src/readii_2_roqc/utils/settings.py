from readii.utils import logger
import numpy as np

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


def check_setting_superset(setting_list: set, 
                           setting_request: set | list | None
                           ) -> bool:
    """
    Check if the requested settings are all in the global settings list for READII.

    Parameters
    ----------
    setting_list: set
        Set of allowed values for a READII setting. Should be set using the global sets in this file.
    setting_request: set |
    """
    if setting_request == [] or setting_request is None:
        return True
    if not setting_list.issuperset(setting_request):
        message = f"Requested settings ({setting_request}) is not a subset of the allowed settings ({setting_list})."
        logger.error(message)
        raise ValueError(message)

    else:
        return True


def get_readii_settings(dataset_config: dict) -> tuple[list, list, list]:
    """Extract READII settings from a configuration dictionary.
    
    Parameters
    ----------
    dataset_config : dict
        Configuration dictionary read in with `loadImageDatasetConfig` containing READII settings
    
    Returns
    -------
    tuple
        A tuple containing:
        - regions: list of regions to process
        - permutations: list of permutations to apply
        - crop: list of crop settings
    """
    readii_config = dataset_config['READII']
    if 'IMAGE_TYPES' not in readii_config:
        message = "READII configuration must contain 'IMAGE_TYPES'."
        logger.error(message)
        raise KeyError(message)
    
    regions = readii_config['IMAGE_TYPES']['regions']
    # Confirm requested regions are available settings
    assert check_setting_superset(REGIONS, regions)
        
    permutations = readii_config['IMAGE_TYPES']['permutations']
    # Confirm requested permutations are available settings
    assert check_setting_superset(PERMUTATIONS, permutations)

    crop = readii_config['IMAGE_TYPES']['crop']
    if crop is not None and crop != []:
        # Confirm requested crop is an available setting
        assert check_setting_superset(CROP, crop)
        # Get single crop value out of list format
        crop = crop[0]
    else:
        crop = ""

    resize = readii_config['IMAGE_TYPES']['resize']
    if resize is None:
        resize = []
    elif isinstance(resize, list):
        match len(resize):
            case 1: resize = resize[0]
            case 3: resize = resize
            case 0: resize
            case _: 
                message = f"READII resize must be a single int, or list of three ints (e.g. [50, 50, 50]). Current value: {resize}"
                logger.error(message)
                raise TypeError(message)
    elif not isinstance(resize, int):
        message = f"READII resize must be a single int, or list of three values (e.g. [50, 50, 50]). Current value: {resize}"
        logger.error(message)
        raise TypeError(message)

    return regions, permutations, crop, resize