from damply import dirs
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import yaml

from readii_2_roqc.feature_extraction.extract import extract_sample_features, metadata_setup
from readii_2_roqc.feature_extraction.index import get_mit_extraction_index


def extract_one_pyradiomics(
        sample_data: pd.Series,
        extraction_settings: str,
        nifti_images_dir_path: str,
        output_features_path: str,
        overwrite: bool = False
) -> dict[str, str]:
    """Extract features from a single scan and mask pair using pyradiomics."""
    # Set up metadata for the sample
    metadata = metadata_setup(sample_data)

    sample_feature_vector = extract_sample_features(sample_data=sample_data,
                                                    method='pyradiomics',
                                                    settings=extraction_settings,
                                                    data_dir=nifti_images_dir_path.parent,
                                                    feature_dir=output_features_path,
                                                    overwrite=overwrite)
    # Add sample metadata to the feature vector for this sample
    metadata.update(sample_feature_vector)

    return metadata



dataset = "HEAD-NECK-RADIOMICS-HN1"

config_file_path = dirs.CONFIG / 'datasets' / f'{dataset}.yaml'
nifti_images_dir_path = dirs.PROCDATA / dataset / "nifti_images" 
output_features_path = dirs.PROCDATA / dataset / "features"
extraction_settings = dirs.CONFIG / 'pyradiomics' / 'linear_all_images_features.yaml'

n_jobs = -1

# Load dataset configuration
with config_file_path.open("r") as f:
    dataset_config = yaml.safe_load(f)

# Set up and save out the index file to run feature extraction with
mit_simple_index_path = nifti_images_dir_path / f"{nifti_images_dir_path.stem}_index-simple.csv"
extraction_index = get_mit_extraction_index(dataset_config, mit_simple_index_path)
if 'DataSource' not in extraction_index.columns:
    extraction_index['DataSource'] = 'PMCC'

output_features_path.mkdir(parents=True, exist_ok=True)
extraction_index.to_csv(output_features_path / f"pyradiomics_{dataset}_index.csv", index=False)

# initialize empty list to hold each feature set per sample
feature_vector = []

# Sequentially extract features
feature_vectors = Parallel(n_jobs=n_jobs)(
    delayed(extract_one_pyradiomics)(
        sample_data=sample_data,
        extraction_settings=extraction_settings,
        nifti_images_dir_path=nifti_images_dir_path,
        output_features_path=output_features_path,
        overwrite=True
    )
    for _, sample_data in tqdm(
        extraction_index.iterrows(),
        desc=f"Extracting pyradiomics features",
        total=len(extraction_index)
    )
)

# Concatenate all the feature vectors
features_df = pd.DataFrame.from_dict(feature_vectors)
# # Save out the features to a single CSV
features_df.to_csv(output_features_path / f"pyradiomics_{dataset}_features.csv", index=False)