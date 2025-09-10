# This script merges radiomics features extracted using PyRadiomics from multiple samples into a single CSV file.

import os
import pandas as pd

base_dir = "/home/bhkuser2/bhklab/radiomics/PublicDatasets/procdata/Lung/TCIA_NSCLC-Radiomics/features/pyradiomics/pyradiomics_h4h_all_images_features"

output_csv = os.path.join(base_dir, 'all_features.csv')

rows = []
folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]

for folder in folders:
    csv_path = os.path.join(base_dir, folder, 'GTV', 'original_full_features.csv')
    if os.path.exists(csv_path):
        # Try semicolon separator first
        df = pd.read_csv(csv_path, sep=';', index_col=0, header=None)
        print(f"Processing {csv_path}, shape: {df.shape}")
        
        # Convert the series to a dictionary and add sample ID
        row = df.iloc[:, 0].to_dict()
        row['SampleID'] = folder
        rows.append(row)

merged_df = pd.DataFrame(rows)
merged_df.to_csv(output_csv, index=False)
print(f"Merged {len(rows)} samples into {output_csv}")
print(f"Final merged DataFrame shape: {merged_df.shape}")