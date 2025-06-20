from pathlib import Path
import SimpleITK as sitk
import pandas as pd
from collections import OrderedDict

from radiomics import featureextractor, setVerbosity
from readii.utils import logger

def sample_feature_writer(feature_vector : pd.Series,
                          metadata : dict[str, str]):
    return

def metadata_setup(metadata : dict[str, str] | pd.Series) -> OrderedDict:
    return


def pyradiomics_extract(settings: Path | str,
                        image: sitk.Image,
                        mask: sitk.Image,
                        metadata : OrderedDict | dict[str, str] | pd.Series | None = None,
                        overwrite : bool = False
                        ) -> pd.Series:
    """Extract PyRadiomic features from an image and mask pair based on configuration from settings file.
    
    Parameters
    ----------
    
    Returns
    -------

    """
    # Set PyRadiomics verbosity to critical only
    setVerbosity(50)

     # Confirm settings file exists
    try:
        assert Path(settings).exists()
    except AssertionError as e:
        logger.error(f"Settings file for PyRadiomics feature extraction at {settings} does not exist.")
        raise e

    # Convert settings Path to string for pyradiomics to read it
    if isinstance(settings, Path):
        settings = str(settings)

    
    
    try:
        # Set up PyRadiomics feature extractor with provided settings file (expects a string, not a pathlib Path)
        extractor = featureextractor.RadiomicsFeatureExtractor(settings)

        sample_feature_vector = pd.Series(extractor.execute(image, mask))

    except Exception as e:
        print(f"Feature extraction failed for this sample: {e}")



    return