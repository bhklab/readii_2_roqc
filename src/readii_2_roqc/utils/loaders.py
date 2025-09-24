from damply import dirs
from readii.io.loaders import loadImageDatasetConfig
from readii.process.config import get_full_data_name
from readii.utils import logger
from readii.image_processing import alignImages, flattenImage

import SimpleITK as sitk


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



def load_image_and_mask(image_path, mask_path = None):
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