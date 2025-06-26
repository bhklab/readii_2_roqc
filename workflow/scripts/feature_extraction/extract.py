from pathlib import Path
import SimpleITK as sitk
import pandas as pd
from collections import OrderedDict

from radiomics import featureextractor, setVerbosity
from readii.utils import logger

from damply import dirs

def sample_feature_writer(feature_vector : pd.Series,
                          metadata : dict[str, str],
                          extraction_settings_name : str):
    
    output_path = dirs.PROCDATA / "features" /  extraction_settings_name 



    return

def metadata_setup(metadata : dict[str, str] | pd.Series) -> OrderedDict:
    """Convert metadata to an OrderedDict.
    
    Parameters
    ----------
    metadata : dict[str, str] | pd.Series
        Metadata to converted into an OrderedDict. Can be a dict or pd.Series.
    
    Returns
    -------
    od_metadata : OrderedDict
        Metadata as an OrderedDict object.
    
    Raises
    ------
    NotImplementedError
        If metadata is not of type dict or pd.Series
    """
    match metadata:
        case dict():
            od_metdata = OrderedDict(metadata)
        case pd.Series():
            od_metadata = metadata.to_dict(into=OrderedDict)
        case _:
            message = f"Metadata must be of dict or pd.Series type. Other types not handled yet. Metadata you've passed is of type {type(metadata)}."
            logger.debug(message)
            raise NotImplementedError(message)

    return od_metadata


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

        sample_feature_vector = extractor.execute(image, mask)

    except Exception as e:
        print(f"Feature extraction failed for this sample: {e}")

    # Set up metadata as an OrderedDict to be combined with the features
    if not isinstance(metadata, OrderedDict):
        metadata = metadata_setup(metadata)
    
    metadata.update(sample_feature_vector)

    
    return metadata