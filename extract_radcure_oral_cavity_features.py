from readii_2_roqc.feature_extraction.extract import pyradiomics_extract
import pandas as pd
from tqdm import tqdm
import SimpleITK as sitk


dataset_index = pd.read_csv('/home/bhkuser/bhklab/katy/readii_2_roqc/data/procdata/TCIA_RADCURE/features/pyradiomics/pyradiomics_RADCURE_GTVp_oral_cavity_index.csv')

method = 'pyradiomics'
settings = '/home/bhkuser/bhklab/katy/readii_2_roqc/config/pyradiomics/pyradiomics_h4h_all_images_features.yaml'

feature_vector = []

# Sequentially extract features
for _, sample_data in tqdm(
    dataset_index.iterrows(),
    desc=f"Extracting {method} features",
    total=len(dataset_index)
):
    image = sitk.ReadImage(sample_data['Image'])
    mask = sitk.ReadImage(sample_data['Mask'])
    mask.SetOrigin(image.GetOrigin())

    sample_feature_vector = pyradiomics_extract(settings = settings,
                                                image = image,
                                                mask = mask,
                                                metadata = sample_data,
                                                overwrite=False)
    # print(sample_feature_vector)
    feature_vector.append(sample_feature_vector)


features_df = pd.DataFrame.from_dict(feature_vector)
features_df.to_csv('/home/bhkuser/bhklab/katy/readii_2_roqc/data/results/TCIA_RADCURE/features/pyradiomics/pyradiomics_RADCURE_GTVp_oral_cavity_features.csv')
