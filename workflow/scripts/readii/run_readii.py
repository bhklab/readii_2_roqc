import pandas as pd
from pathlib import Path
import numpy as np

import radiomics
from radiomics import featureextractor, setVerbosity

import SimpleITK as sitk

from tqdm import tqdm
from joblib import Parallel, delayed
from collections import OrderedDict
import itertools

from readii.negative_controls_refactor.manager import NegativeControlManager

# from readii.utils import logger

def flattenImage(image: sitk.Image) -> sitk.Image:
    """Remove axes of image with size one. (ex. shape is [1, 100, 256, 256])

    Parameters
    ----------
    image : sitk.Image
        Image to remove axes with size one.

    Returns
    -------
    sitk.Image
        image with axes of length one removed.
    """
    imageArr = sitk.GetArrayFromImage(image)

    imageArr = np.squeeze(imageArr)

    return sitk.GetImageFromArray(imageArr)


def pyradiomics_extraction(extractor: radiomics.featureextractor,
                           image: sitk.Image,
                           mask: sitk.Image,
                           sample_info: pd.Series,
                           sample_dir_path: Path,
                           region: str | None = None,
                           transform: str | None = None,
                           overwrite: bool = False
                           ) -> pd.Series:
    """Perform PyRadiomic feature extraction on a single image and mask pair.

    Parameters
    ----------
    extractor : radiomics.featureextractor
        PyRadiomics feature extractor object set up with parameters
    image : sitk.Image
        Main 3D image to perform feature extraction on.
    mask : sitk.Image
        3D binary mask specifying region to perform feature extraction on. Region of interest (ROI) voxels assumed to be 1.
    sample_info : pd.Series
        Info pertaining to the image and mask combo, should contain three values:
            1. ID - unique identifier for the image (e.g. sample ID)
            2. Image - path to the image file
            3. Mask - path to the mask file
    sample_dir_path : Path
        Path to directory to create a sample directory in to store the output extracted feature file for this sample.
        Example: "data/procdata/TCIA_NSCLC-Radiomics/pyradiomics_original_all_features/"
    region : str | None, default=None
        If image has had a negative control applied, region specifies where the transformation was applied. Used in naming the output file.
    transform : str | None, default=None
        If image has had a negative control applied, transform specifies what transformation was applied. Used in naming the output file.
    overwrite : bool, default=False
        Whether to extract features if feature file already exists at in the sample_dir_path for this sample.
        If file exists and `overwrite == False`, will print a message stating features were already extracted for this sample.

    Returns
    -------
    sample_result_series : pd.Series
        Series containing 
            * the ID, Mask, and Image data from `sample_info`
            * PyRadiomics diagnostics outputs
            * PyRadiomics feature outputs
    """
    # Set PyRadiomics verbosity to critical only
    setVerbosity(50)

    # check if file already exists
    if region and transform:
        sample_result_file_name = f"{region}_{transform}_features.csv"
    else:
        sample_result_file_name = f"full_original_features.csv"
    
    # Set up sample directory to save output files in
    complete_out_path = sample_dir_path / sample_result_file_name 
    if complete_out_path.exists() and (not overwrite):
        message = f"Features already extracted for: {complete_out_path.stem}"
        print(message)
        # logger.info(message)
        return
    
    else:
        try:
            # Perform feature extraction on the ROI specified by the mask on the image
            sample_feature_vector = pd.Series(extractor.execute(image, mask))
        except Exception as e:
            print(f"Feature extraction failed for {sample_info.ID} {region} {transform}: {e}")
        
        # Start the result dictionary with the sample info (must convert within this loop to create new result dict every time)
        sample_result_dict = sample_info.to_dict(into=OrderedDict)
        # Append the extracted features to the result dictionary
        sample_result_dict.update(sample_feature_vector)

        # Save out sample info and extracted features as a csv
        sample_result_series = pd.Series(sample_result_dict).to_csv(complete_out_path)

    return sample_result_series



def combine_feature_results(procdata_path: Path,
                            results_path: Path,
                            extraction_params: Path,
                            nc_manager : NegativeControlManager | None = None,
                            ) -> list[Path]:
    """Combine sample-wise feature extraction output files for an entire datasets for the default image type and those in a NegativeControlManager.
    If no feature files exist for an image type, an empty .csv file is writen and an error is logged.

    Parameters
    ----------
    procdata_path : Path
        Path to directory where radiomic features were saved. (e.g. what was passed to `extra_features`)
    results_path : Path
        Path to directory to save the combine radiomic feature files. 
    extraction_params : Path
        Path to extraction parameter configuration file. Used to name output directory combined feature files will be saved in.
    nc_manager : NegativeControlManager | None, default = None
        READII Negative control manager used on images for feature extraction. Used to collect all the sample feature sets for each image type.
    
    Returns
    -------
    combined_feature_path_list : list[Path]
        List of saved output paths.
    """
    # Setup input directory to search through
    samplewise_feature_dir_path = procdata_path / Path(extraction_params).stem
    # Setup output directory to save combined feature files to
    output_dir_path = results_path / Path(extraction_params).stem
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize strategy list with default image type (no negative control applied)
    strategy_list = ["full_original"]

    # Initialize list to store output file paths in
    combined_feature_path_list = []

    # add negative controls to the strategy list if passed in
    if nc_manager is not None:
        for strategy_combo in nc_manager.strategy_products:
            strategy_list.append(f"{strategy_combo[1].region_name}_{strategy_combo[0].negative_control_name}")

    # Search for all the sample feature files for each image type
    for image_type in strategy_list:
        # Regex for directory search
        filename_pattern = f"*/{image_type}_features.csv"

        # List of sample feature files for this image type
        feature_file_list = sorted(samplewise_feature_dir_path.rglob(filename_pattern))

        # Set up output file path for this image type
        combined_feature_path = output_dir_path / f"{image_type}_features.csv"

        # Generator
        empty_files = [] 
        def non_empty_dfs(file_list):
            for file in file_list:
                try:
                    df = pd.read_csv(file, index_col=0)
                    if not df.empty:
                        empty_files.append(file)
                        yield df.T
                except pd.errors.EmptyDataError:
                    pass

        try:
            # Find all non-empty feature dataframes in the globbed list and concatenate them
            all_sample_features = pd.concat(non_empty_dfs(feature_file_list))
            # Sort the dataframes by the sample ID column
            all_sample_features.sort_values(by="ID", inplace=True)
            # Save out the combined feature dataframe
            all_sample_features.to_csv(combined_feature_path, index=False)

        except ValueError:
            # Handle case where all dataframes are empty
            # logger.error("No non-empty dataframes found.")
            # write empty file to the output file
            with open(combined_feature_path, "w") as f:
                # write an empty file
                f.write("")
            # logger.error(f"Empty file written to {combined_feature_path}")

        # Store the output file path to return
        combined_feature_path_list.append(combined_feature_path)
    
    return combined_feature_path_list



def extract_features(dataset_index : pd.DataFrame,
                     procdata_path : Path,
                     pyrad_params : str,
                     nc_manager : NegativeControlManager | None = None,
                     parallel : bool = False,
                     overwrite : bool = False,
                     ) -> tuple[list[pd.Series], Path]:
    """Extract radiomic features for each sample in a given dataset. Optionally apply negative controls and run in parallel.

    Parameters
    ----------
    dataset_index : pd.DataFrame
        Table of samples to process. Must contain three columns labelled:
            1. ID - unique identifier for the image (e.g. sample ID)
            2. Image - path to the image file
            3. Mask - path to the mask file
    procdata_path : Path
        Path to directory to save sample feature outputs to. Will create a directory here named the same as the extraction parameter file.
    pyrad_params : str
        String of the path to the PyRadiomics configuration file to use for feature extraction.
    nc_manager : NegativeControlManager | None, default=None
        READII Negative control manager to be used on images for feature extraction if provided. 
    parallel: bool, default=False
        Whether to run feature extraction in a parallel fashion. Note that this currently fails when running non-ROI negative controls.
    overwrite : bool, default=False
        Whether to overwrite existing feature files if present. 

    Returns
    -------
    sample_results : list[pd.Series]
        List of sample feature extraction results. Each entry is a pandas Series containing the sample ID, image path, mask path, and extracted features.
    dataset_feature_dir : Path
        Path to the directory where the sample feature files were saved. This is the same as `procdata_path` with the extraction parameter file name appended.    
    """
    # Set up PyRadiomics feature extractor
    extractor = featureextractor.RadiomicsFeatureExtractor(pyrad_params)
    dataset_feature_dir = procdata_path / Path(pyrad_params).stem

    for idx, sample_row in tqdm(dataset_index.iterrows(), total=len(dataset_index)):
        # Set up output dir for this sample's features
        roi_name = Path(Path(sample_row.Mask).stem).stem
        sample_feature_dir = dataset_feature_dir / sample_row.ID / roi_name
        sample_feature_dir.mkdir(parents=True, exist_ok=True)

        # Load image and ROI mask for this sample
        sample_image = flattenImage(sitk.ReadImage(sample_row.Image))
        sample_mask = flattenImage(sitk.ReadImage(sample_row.Mask))

        # Set up image types to iterate over, including negative controls if provided
        if nc_manager:
            image_types = itertools.chain([(sample_image, "original", "full")], 
                                          nc_manager.apply(sample_image, sample_mask))
        else:
            # If no negative control manager is provided, just use the original image and mask
            image_types = itertools.chain([(sample_image, "original", "full")])

        # Feature Extraction call in parallel or sequentially
        if not parallel:
            sample_results = [pyradiomics_extraction(
                                        extractor=extractor,
                                        image=image,
                                        mask=sample_mask,
                                        sample_info=sample_row,
                                        sample_dir_path=sample_feature_dir,
                                        region=region,
                                        transform=transform,
                                        overwrite=overwrite
                                        )
                                        for image, transform, region in image_types
                                    ]
        else:
            sample_results = Parallel(n_jobs=-1, require="sharedmem")(
                                delayed(pyradiomics_extraction)(
                                    extractor=extractor,
                                    image=neg_image,
                                    mask=sample_mask,
                                    sample_info=sample_row,
                                    sample_dir_path=sample_feature_dir,
                                    region=region,
                                    transform=transform,
                                    overwrite=overwrite
                                )
                                for neg_image, transform, region in image_types
                            )

    return sample_results, dataset_feature_dir



def main(dataset_index:pd.DataFrame,
         pyrad_params:str,
         procdata_path:Path,
         results_path:Path,
         regions: list[str] | None = None,  
         transforms: list[str] | None = None,
         overwrite:bool = False,
         parallel:bool = True,
         seed:int = 10        
        ) -> tuple[Path, pd.DataFrame]:
    """Main function to extract features from a dataset using PyRadiomics, apply negative controls if specified, and consolidate results.
    This function is designed to be run as a script, and will extract features from the dataset specified in the `dataset_index` parameter.
    It will create a directory for the extracted features in the `procdata_path` directory, and save the results in the `results_path` directory.

    Parameters
    ----------
    dataset_index : pd.DataFrame
        Table of samples to process. Must contain three columns labelled:
            1. ID - unique identifier for the image (e.g. sample ID)
            2. Image - path to the image file
            3. Mask - path to the mask file
    pyrad_params : str
        String of the path to the PyRadiomics configuration file to use for feature extraction.
    procdata_path : Path
        Path to directory to save sample feature outputs to. Will create a directory here named the same as the extraction parameter file.
    results_path : Path
        Path to directory to save combined feature outputs to. Will create a directory here named the same as the extraction parameter file.
    regions : list[str] | None, default=None
        List of regions to apply negative controls to. If empty, no negative controls will be applied.
    transforms : list[str] | None, default=None
        List of transforms to apply to the regions. If empty, no negative controls will be applied.
    overwrite : bool, default=False
        Whether to overwrite existing feature files if present.
    parallel : bool, default=True
        Whether to run feature extraction in parallel. Note that this currently fails when running non-ROI negative controls.
    seed : int, default=10
        Random seed to use for negative control generation. This is used to ensure that the same negative controls are generated each time the script is run.

    Returns
    -------
    feature_path : Path
        Path to the directory where the combined feature files were saved. This is the same as `results_path` with the extraction parameter file name appended.
    sample_results : pd.DataFrame
        DataFrame containing the sample feature extraction results. Each entry is a pandas Series containing the sample ID, image path, mask path, and extracted features.
    """    
    # Set up negative control generator based on inputs
    if regions and transforms:
        manager = NegativeControlManager.from_strings(
            negative_control_types=transforms,
            region_types=regions,
            random_seed=seed
        )
    else:
        manager = None

    # Extract features
    sample_results, dataset_feature_dir = extract_features(dataset_index=dataset_index,
                                                            procdata_path=procdata_path,
                                                            nc_manager = manager,
                                                            pyrad_params = pyrad_params,
                                                            parallel=parallel,
                                                            overwrite=overwrite,
                                                            )

    # Collect all results 
    feature_path = combine_feature_results(nc_manager=manager,
                                           procdata_path=procdata_path,
                                           results_path=results_path,
                                           extraction_params= pyrad_params)

    return feature_path, sample_results



if __name__ == "__main__":
    RANDOM_SEED = 10

    # general data directory path setup
    DATA_DIR_PATH = Path("../../data")
    RAW_DATA_PATH = DATA_DIR_PATH / "rawdata"
    PROC_DATA_PATH = DATA_DIR_PATH / "procdata"
    RESULTS_DATA_PATH = DATA_DIR_PATH / "results"

    # specific dataset path setup
    DATA_SOURCE = "TCIA"
    DATASET_NAME = "HEAD-NECK-RADIOMICS-HN1"
    
    dataset = f"{DATA_SOURCE}_{DATASET_NAME}"
    dataset_index_path = PROC_DATA_PATH / dataset / f"pyrad_{dataset}_index.csv"
    dataset_index = pd.read_csv(dataset_index_path)

    # PYRADIOMICS CONFIGURATION
    parameter_file_path = "../../config/pyradiomics/pyradiomics_original_all_features.yaml"


    sample_results = main(dataset_index = dataset_index,
                          pyrad_params = parameter_file_path,
                          procdata_path = PROC_DATA_PATH / dataset,
                          results_path = RESULTS_DATA_PATH / dataset,
                          regions = ["full", "roi", "non_roi"],
                          transforms = ["randomized", "shuffled", "sampled"],
                          overwrite = False,
                          parallel = False,
                          seed = RANDOM_SEED)
    


    
