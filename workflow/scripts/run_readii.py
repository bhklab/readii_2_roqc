import pandas as pd
from pathlib import Path

import radiomics
from radiomics import featureextractor, setVerbosity

import SimpleITK as sitk

from tqdm import tqdm
from joblib import Parallel, delayed
from collections import OrderedDict
import itertools

from readii.negative_controls_refactor.manager import NegativeControlManager
from readii.image_processing import flattenImage

from readii.utils import logger


def pyradiomics_extraction(extractor:radiomics.featureextractor,
                           image:sitk.Image,
                           mask:sitk.Image,
                           sample_info:pd.Series,
                           sample_dir_path:Path,
                           region:str = None,
                           transform:str = None,
                           overwrite:bool = False
                           ):
    # check if file already exists
    if region and transform:
        sample_result_file_name = f"{sample_info.ID}_{region}_{transform}_features.csv"
    else:
        sample_result_file_name = f"{sample_info.ID}_full_original_features.csv"
    
    complete_out_path = sample_dir_path / sample_result_file_name
    if complete_out_path.exists() and (not overwrite):
        print(f"Features already extracted for: {complete_out_path.stem}")
        return
    
    else:
        try:
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


def combine_feature_results(nc_manager:NegativeControlManager,
                             samplewise_feature_dir_path:Path,
                             output_dir_path:Path,
                             ):
    
    strategy_list = ["full_original"]
    for strategy_combo in nc_manager.strategy_products:
        strategy_list.append(f"{strategy_combo[1].region_name}_{strategy_combo[0].negative_control_name}")

    for negative_control in strategy_list:
        feature_file_list = sorted(samplewise_feature_dir_path.rglob(f"*[0-9]_{negative_control}_features.csv"))

        combined_feature_path = output_dir_path / f"{negative_control}_features.csv"
        combined_feature_path.parent.mkdir(parents=True, exist_ok=True)

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
            all_sample_features = pd.concat(non_empty_dfs(feature_file_list))
            all_sample_features.sort_values(by="ID", inplace=True)
            all_sample_features.to_csv(combined_feature_path, index=False)
        except ValueError:
            # Handle case where all dataframes are empty
            logger.error("No non-empty dataframes found.")
            # write empty file to the output file
            with open(combined_feature_path, "w") as f:
                # write an empty file
                f.write("")
            logger.error(f"Empty file written to {combined_feature_path}")
    
    return combined_feature_path



def main(dataset_index:pd.DataFrame,
         pyrad_params:str,
         procdata_path:Path,
         results_path:Path,
         regions:list[str] = ["roi", "non_roi", "full"],
         transforms:list[str] = ["randomized", "shuffled", "sampled"],
         overwrite:bool = False,
         parallel:bool = True,
         seed:int = 10        
):
    # Set PyRadiomics verbosity to critical only
    setVerbosity(50)
    
    # Set up PyRadiomics feature extractor
    extractor = featureextractor.RadiomicsFeatureExtractor(pyrad_params)

    # Set up negative control generator based on inputs
    if regions and transforms:
        manager = NegativeControlManager.from_strings(
            negative_control_types=transforms,
            region_types=regions,
            random_seed=seed
        )

    for idx, sample_row in tqdm(dataset_index.iterrows(), total=len(dataset_index)):
        # Set up output dir for this sample's features
        sample_feature_dir = procdata_path / Path(pyrad_params).stem / sample_row.ID
        sample_feature_dir.mkdir(parents=True, exist_ok=True)

        # Load image and ROI mask for this sample
        sample_image = flattenImage(sitk.ReadImage(sample_row.Image))
        sample_mask = flattenImage(sitk.ReadImage(sample_row.Mask))

        if not parallel:
            sample_results = [pyradiomics_extraction(
                                        extractor=extractor,
                                        image=neg_image,
                                        mask=sample_mask,
                                        sample_info=sample_row,
                                        sample_dir_path=sample_feature_dir,
                                        region=region,
                                        transform=transform,
                                        overwrite=overwrite
                                        )
                                        for neg_image, transform, region in 
                                        itertools.chain([(sample_image, "original", "full")], 
                                                          manager.apply(sample_image, sample_mask))
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
                                for neg_image, transform, region in 
                                itertools.chain([(sample_image, "original", "full")], 
                                                  manager.apply(sample_image, sample_mask))
                            )

    # Collect all results 
    samplewise_feature_dir_path = procdata_path / Path(pyrad_params).stem
    dataset_features_dir_path = results_path / Path(pyrad_params).stem
    feature_path = combine_feature_results(nc_manager=manager,
                                           samplewise_feature_dir_path=samplewise_feature_dir_path,
                                           output_dir_path=dataset_features_dir_path)

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
    DATASET_NAME = "NSCLC-Radiomics"
    
    dataset = f"{DATA_SOURCE}_{DATASET_NAME}"
    dataset_index_path = PROC_DATA_PATH / dataset / f"pyrad_{dataset}_index.csv"
    dataset_index = pd.read_csv(dataset_index_path)

    # PYRADIOMICS CONFIGURATION
    parameter_file_path = "../../config/pyradiomics/pyradiomics_original_all_features.yaml"


    sample_results = main(dataset_index = dataset_index,
                          pyrad_params = parameter_file_path,
                          procdata_path = PROC_DATA_PATH / dataset,
                          results_path = RESULTS_DATA_PATH / dataset,
                          regions = ["full", "roi"],
                          transforms = ["shuffled", "sampled", "randomized"],
                          overwrite = False,
                          parallel = True,
                          seed = RANDOM_SEED)
    


    
