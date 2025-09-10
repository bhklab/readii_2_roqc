# This script synchronizes and merges radiomics feature datasets from FMCIB, PyRadiomics, and LIFEx.

import pandas as pd
import os
import seaborn as sns
import re
import matplotlib.pyplot as plt


def extract_fmcib_id(image_path):
  # Example: .../HN1006_1.nii.gz -> HN1006_1
  return os.path.splitext(os.path.basename(image_path))[0].replace('.nii', '')

def extract_pyradiomics_id(sample_id):
  # Example: HN1006_0001 -> HN1006_0001
  return sample_id

def extract_lifex_id(series_path):
  # Example: .../HN1006_0001/CT_25815574/CT.nii.gz -> HN1006_0001
  parts = series_path.split(os.sep)
  return parts[-3] if len(parts) > 1 else None

def standardize_sync_id(sync_id):
  # Convert LUNG1006_1 to LUNG1006_0001 format
  match = re.match(r'(LUNG1-\d+)_(\d+)', sync_id)
  prefix, num = match.groups()
  return f"{prefix}_{int(num):04d}"

def sync_data(fmcib_csv, pyradiomics_csv, lifex_csv):
  fmcib_df = pd.read_csv(fmcib_csv)
  pyradiomics_df = pd.read_csv(pyradiomics_csv)
  lifex_df = pd.read_csv(lifex_csv)

  fmcib_df['sync_id'] = fmcib_df['image_path'].apply(extract_fmcib_id)
  fmcib_df['sync_id'] = fmcib_df['sync_id'].apply(standardize_sync_id)
  pyradiomics_df['sync_id'] = pyradiomics_df['SampleID'].apply(extract_pyradiomics_id)
  lifex_df['sync_id'] = lifex_df['INFO_SeriesPath'].apply(extract_lifex_id)

  # add data name as suffix to all columns except 'sync_id'
  fmcib_df = fmcib_df.rename(columns=lambda x: f"{x}_fmcib" if x != 'sync_id' else x)
  pyradiomics_df = pyradiomics_df.rename(columns=lambda x: f"{x}_pyradiomics" if x != 'sync_id' else x)
  lifex_df = lifex_df.rename(columns=lambda x: f"{x}_lifex" if x != 'sync_id' else x)

  # Merge on 'sync_id' to align rows
  merged = fmcib_df.merge(pyradiomics_df, on='sync_id')
  merged = merged.merge(lifex_df, on='sync_id')
  
  # keep only sync_id and numeric columns
  numeric_columns = merged.select_dtypes(include=['number']).columns.tolist()
  numeric_columns.append('sync_id')
  merged = merged[numeric_columns]
  merged.rename(columns={'sync_id': 'ID'}, inplace=True)
  
  merged.to_csv('merged_data_nsclc-radiomics.csv', index=False)

  return

if __name__ == "__main__":
  
  # #HN1 
  # pyradiomics = "/home/bhkuser2/bhklab/radiomics/PublicDatasets/procdata/HeadNeck/TCIA_HEAD-NECK-RADIOMICS-HN1/features/pyradiomics/all_features.csv"
  # fmcib = "/home/bhkuser2/bhklab/radiomics/Projects/readii-fmcib/results/TCIA_HEAD-NECK-RADIOMICS-HN1/fmcib_features/cropped_bbox/fmcib_features_original.csv"
  # lifex = "/home/bhkuser2/bhklab/radiomics/PublicDatasets/procdata/HeadNeck/TCIA_HEAD-NECK-RADIOMICS-HN1/features/lifex/lifex_HEAD-NECK-RADIOMICS-HN1_features.csv"

  # NSCLC
  pyradiomics = "/home/bhkuser2/bhklab/radiomics/PublicDatasets/procdata/Lung/TCIA_NSCLC-Radiomics/features/pyradiomics/pyradiomics_h4h_all_images_features/all_features.csv"
  fmcib = "/home/bhkuser2/bhklab/radiomics/Projects/readii-fmcib/results/TCIA_NSCLC-Radiomics/fmcib_features/cropped_bbox/fmcib_features_original.csv"
  lifex = "/home/bhkuser2/bhklab/radiomics/PublicDatasets/procdata/Lung/TCIA_NSCLC-Radiomics/features/lifex/lifex_NSCLC-Radiomics_features.csv"

  sync_data(fmcib, pyradiomics, lifex)
