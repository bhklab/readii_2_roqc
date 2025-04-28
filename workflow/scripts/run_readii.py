import pandas as pd
from pathlib import Path
from radiomics import featureextractor, setVerbosity
import SimpleITK as sitk


from readii.negative_controls_refactor.manager import NegativeControlManager


def pyradiomics_extraction(extractor:radiomics.featureextractor,
                           image:sitk.Image,
                           mask:sitk.Image,
                           ):
    return



def main(dataset_index:pd.DataFrame,
         pyrad_params:str,
         procdata_path:Path,
         results_path:Path,
         regions:list[str],
         transforms:list[str],
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

    
    
    return