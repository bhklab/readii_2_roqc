from damply import dirs
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
import yaml

from readii_2_roqc.feature_extraction.extract import extract_sample_features
from readii_2_roqc.feature_extraction.index import get_mit_extraction_index

dataset = "cbct_dataset"

config_file_path = dirs.CONFIG / 'datasets' / f'{dataset}.yaml'
nifti_images_dir_path = dirs.PROCDATA / dataset / "nifti_images" 
output_features_path = dirs.PROCDATA / dataset / "features"
extraction_settings = dirs.CONFIG / 'pyradiomics' / 'linear_all_images_features.yaml'

# Load dataset configuration
with config_file_path.open("r") as f:
    dataset_config = yaml.safe_load(f)

# Set up and save out the index file to run feature extraction with
mit_simple_index_path = nifti_images_dir_path / f"{dataset}_index-simple.csv"
extraction_index = get_mit_extraction_index(dataset_config, mit_simple_index_path)
if 'DataSource' not in extraction_index.columns:
    extraction_index['DataSource'] = 'PMCC'

extraction_index.to_csv(output_features_path / f"pyradiomics_{dataset}_index.csv", index=False)

# initialize empty list to hold each feature set per sample
feature_vector = []

# Sequentially extract features
for _, sample_data in tqdm(
    extraction_index.iterrows(),
    desc=f"Extracting pyradiomics features",
    total=len(extraction_index)
):
    image = sitk.ReadImage(nifti_images_dir_path / sample_data['Image'])
    mask = sitk.ReadImage(nifti_images_dir_path / sample_data['Mask'])
    mask.SetOrigin(image.GetOrigin())

  
    sample_feature_vector = extract_sample_features(sample_data=sample_data,
                                                        method='pyradiomics',
                                                        settings=extraction_settings,
                                                        overwrite=False)

    feature_vector.append(sample_feature_vector)

# Concatenate all the feature vectors
features_df = pd.DataFrame.from_dict(feature_vector)
# Save out the features to a single CSV
features_df.to_csv(output_features_path / f"pyradiomics_{dataset}_features.csv")