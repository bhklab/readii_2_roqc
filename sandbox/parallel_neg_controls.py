import itertools
from pathlib import Path

import click
import pandas as pd
import SimpleITK as sitk
from damply import dirs
from imgtools.io.writers.nifti_writer import NIFTIWriter, NiftiWriterIOError
from tqdm import tqdm

from readii.image_processing import alignImages, flattenImage
from readii.io.loaders import loadImageDatasetConfig
from readii.negative_controls_refactor import NegativeControlManager
from readii.process.config import get_full_data_name
from readii.utils import logger
from readii_2_roqc.utils.metadata import get_masked_image_metadata
from readii_2_roqc.readii.make_negative_controls import get_readii_settings


dataset = 'NSCLC-Radiomics_test'
overwrite = False
seed = 10

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
logger.info(f"Creating negative controls for dataset: {dataset_name}")

# Extract READII settings from config file
regions, permutations, _crop = get_readii_settings(dataset_config)

# Set up negative control manager with settings from config
manager = NegativeControlManager.from_strings(
negative_control_types=permutations,
region_types=regions,
random_seed=seed
)

# Get path to the images output by med-imagetools autopipeline run
mit_images_dir_path = dirs.PROCDATA / full_data_name / 'images' /f'mit_{dataset_name}'

# Load in mit_index file
dataset_index = pd.read_csv(Path(mit_images_dir_path, f'mit_{dataset_name}_index-simple.csv'))

# Get just the rows with the desired image and mask modalities and ROIs specified in the dataset config
image_modality = dataset_config["MIT"]["MODALITIES"]["image"]
mask_modality = dataset_config["MIT"]["MODALITIES"]["mask"]
masked_image_index = get_masked_image_metadata(dataset_index = dataset_index,
                                            dataset_config = dataset_config,
                                            image_modality = image_modality,
                                            mask_modality = mask_modality)

# Set up directory to save out the negative controls
readii_image_dir = mit_images_dir_path.parent / f'readii_{dataset_name}'

# Check for index file existence and overwrite status to determine if continuing to negative control creation
readii_index_file = readii_image_dir / f'readii_{dataset_name}_index.csv'

if readii_index_file.exists() and not overwrite: 
    # Load in readii index and check:
    # 1. if all negative controls requested have been extracted
    # 2. for all of the patients
    readii_index = pd.read_csv(readii_index_file)

    # Get list of patients that have already been processed and what has been requested based on the dataset index
    processed_samples = set(readii_index['PatientID'].to_list())
    requested_samples = set(dataset_index['PatientID'].to_list())

    processed_image_types = {itype for itype in readii_index[['Permutation', 'Region']].itertuples(index=False, name=None)}
    requested_image_types = {itype for itype in itertools.product([permutation.name() for permutation in manager.negative_control_strategies],
                                                                [region.name() for region in manager.region_strategies])}

    # Check if the requested image types are a subset of those already processed
    if requested_image_types.issubset(processed_image_types) and requested_samples.issubset(processed_samples):
        print("Requested negative controls have already been generated for these samples or are listed in the readii index as if they have been. Set overwrite to true if you want to re-process these.")
        exit()

    else:
        print("Some requested negative controls or samples have not been generated yet.")
        
        


print('end of file reached')