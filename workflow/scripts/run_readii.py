import pandas as pd
from pathlib import Path

import radiomics
from radiomics import featureextractor, setVerbosity

import SimpleITK as sitk

from tqdm import tqdm
from joblib import Parallel, delayed
from collections import OrderedDict

from readii.negative_controls_refactor.manager import NegativeControlManager
from readii.image_processing import flattenImage



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

    # Get number of samples for 
    total_sample_count = len(dataset_index)

    for idx, sample_row in tqdm(dataset_index.iterrows(), total=dataset_index.shape[0]):
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
                                        for neg_image, transform, region in manager.apply(sample_image, sample_mask)
                                    ]
        else:
            Parallel(n_jobs=-1, require="sharedmem")(
                delayed(pyradiomics_extraction)(
                    
                )
            )
 

    
    return