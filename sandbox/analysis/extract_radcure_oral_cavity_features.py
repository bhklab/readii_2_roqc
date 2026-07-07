from readii_2_roqc.feature_extraction.extract import extract_sample_features
import pandas as pd
from tqdm import tqdm
import SimpleITK as sitk
from damply import dirs


dataset_index = pd.read_csv('/home/bhkuser/bhklab/katy/readii_2_roqc/data/procdata/PMCC_AutoWATChmAN/features/pyradiomics/original_512_512_n/pyradiomics_AutoWATChmAN_index.csv')
if 'DataSource' not in dataset_index.columns:
    dataset_index['DataSource'] = 'PMCC'


method = 'pyradiomics'
settings = '/home/bhkuser/bhklab/katy/readii_2_roqc/config/pyradiomics/linear_all_images_features.yaml'

feature_vector = []

# Sequentially extract features
for _, sample_data in tqdm(
    dataset_index.iterrows(),
    desc=f"Extracting {method} features",
    total=len(dataset_index)
):
    image = sitk.ReadImage(dirs.PROCDATA / "PMCC_AutoWATChmAN" / "images" / sample_data['Image'])
    mask = sitk.ReadImage(dirs.PROCDATA / "PMCC_AutoWATChmAN" / "images" / sample_data['Mask'])
    mask.SetOrigin(image.GetOrigin())

  
    sample_feature_vector = extract_sample_features(sample_data=sample_data,
                                                        method=method,
                                                        settings=settings,
                                                        overwrite=False)

    feature_vector.append(sample_feature_vector)


features_df = pd.DataFrame.from_dict(feature_vector)
features_df.to_csv('/home/bhkuser/bhklab/katy/readii_2_roqc/data/results/PMCC_AutoWATChmAN/features/pyradiomics/pyradiomics_AutoWATChmAN_features.csv')
