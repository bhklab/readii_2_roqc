import pandas as pd
from damply import dirs
from readii.io.loaders import loadImageDatasetConfig

from readii_2_roqc.feature_extraction.index import get_mit_extraction_index


original_dataset_index_path = dirs.PROCDATA / "PMCC_AutoWATChmAN" / "images" / "slicer_AutoWATChmAN" / "slicer_AutoWATChmAN_index-simple.csv"

config_dir_path = dirs.CONFIG / 'datasets'
dataset_config = loadImageDatasetConfig('AutoWATChmAN', config_dir_path)


extraction_index = get_mit_extraction_index(dataset_config, original_dataset_index_path)

extraction_index.to_csv(dirs.PROCDATA / "PMCC_AutoWATChmAN" / "features" / "pyradiomics" / "original_512_512_n" / "pyradiomics_slicer_AutoWATChmAN_index.csv", index=False)

# print(extraction_index.head())
