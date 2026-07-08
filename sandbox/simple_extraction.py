from damply import dirs
from pathlib import Path
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import yaml
import click

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

@click.command()
@click.argument('config_file', type=Path)
@click.argument('nifti_dir', type=Path)
@click.argument('feature_dir', type=Path)
@click.argument('extract_settings', type=Path)
@click.option('--n_jobs', default=-1, help='Number of parallel jobs to use')
@click.option('--overwrite', is_flag=True, help='Overwrite existing feature files')
def simple_extraction(
    config_file: str,
    nifti_dir: Path,
    feature_dir: Path,
    extract_settings: Path,
    n_jobs: int = -1,
    overwrite: bool = False
)-> None:
    """Extract features from a dataset using pyradiomics.
    
    CONFIG_FILE is the path to the dataset configuration file (YAML format).

    NIFTI_DIR is the path to the directory containing med-imagetools converted NIfTI images. Must contain an index-simple.csv file.

    FEATURE_DIR is the path to the directory where extracted features will be saved.

    EXTRACT_SETTINGS is the path to the extraction settings file (YAML format). 
    """
    # Load dataset configuration
    with config_file.open("r") as f:
        dataset_config = yaml.safe_load(f)

    # Set up and save out the index file to run feature extraction with
    mit_simple_index_path = nifti_dir / f"{nifti_dir.stem}_index-simple.csv"
    extraction_index = get_mit_extraction_index(dataset_config, mit_simple_index_path)
    if 'DataSource' not in extraction_index.columns:
        extraction_index['DataSource'] = 'PMCC'

    # Save out the extraction index to the output features path
    feature_dir.mkdir(parents=True, exist_ok=True)
    extraction_index.to_csv(feature_dir / f"pyradiomics_extraction_index.csv", index=False)

    # extract features in parallel for each sample in the extraction index
    feature_vectors = Parallel(n_jobs=n_jobs)(
        delayed(extract_one_pyradiomics)(
            sample_data=sample_data,
            extraction_settings=extract_settings,
            nifti_images_dir_path=nifti_dir,
            output_features_path=feature_dir,
            overwrite=overwrite
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
    features_df.to_csv(feature_dir / f"pyradiomics_features.csv", index=False)


if __name__ == "__main__":
    simple_extraction()

    # dataset = "HEAD-NECK-RADIOMICS-HN1"

    # config_file_path = dirs.CONFIG / 'datasets' / f'{dataset}.yaml'
    # nifti_images_dir_path = dirs.PROCDATA / dataset / "nifti_images" 
    # output_features_path = dirs.PROCDATA / dataset / "features"
    # extraction_settings = dirs.CONFIG / 'pyradiomics' / 'linear_all_images_features.yaml'

    # n_jobs = -1
    # overwrite = False
