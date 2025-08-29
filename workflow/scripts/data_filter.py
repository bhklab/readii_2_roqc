# This script filters patients older than 65 and younger than 76 with Adenocarcinoma histology
# and excludes those from AMC institution.
import pandas as pd

df = pd.read_csv("/home/bhkuser2/bhklab/shabnam/readii_2_roqc/data/rawdata/TCIA_NSCLC-Radiogenomics/NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv")

df = df[~df['Case ID'].str.startswith('AMC')]
df = df[df['Histology '].str.startswith('Adenocarcinoma')]

df['Age at Histological Diagnosis'] = pd.to_numeric(df['Age at Histological Diagnosis'], errors='coerce')
filtered_df = df[(df['Age at Histological Diagnosis'] > 65) & (df['Age at Histological Diagnosis'] < 76)]

print(f"Number of patients older than 65 and younger than 76: {len(filtered_df)}")
print(filtered_df.head())

#save the filtered DataFrame to a new CSV file
filtered_df.to_csv("/home/bhkuser2/bhklab/shabnam/readii_2_roqc/data/filtered_patients.csv", index=False)
