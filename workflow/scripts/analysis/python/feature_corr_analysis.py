# This script analyzes pairwise correlations between different types of radiomic features

from numpy import full
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering


def feature_type_from_col(col):
  return col.split('_')[-1]

def pairwise_type_correlation(df):
  # Identify types
  types = set(feature_type_from_col(col) for col in df.columns if col != 'ID')
  #remove columns with empty rows
  df = df.dropna(axis=1, how='all')
  results = {}
  for type1 in types:
    for type2 in types:
      if type1 < type2:  # Avoid duplicate pairs and self-pair
        cols1 = [col for col in df.columns if feature_type_from_col(col) == type1]
        cols2 = [col for col in df.columns if feature_type_from_col(col) == type2]
        # if type is pyradiomics, keep the columns starting with 'original'
        # if type1 == 'pyradiomics':
        #   cols1 = [col for col in cols1 if col.startswith('original')]
        # if type2 == 'pyradiomics':
        #   cols2 = [col for col in cols2 if col.startswith('original')]
        # Compute correlation between all pairs
        # corr_matrix = df[cols1].corrwith(df[cols2], axis=0)
        # Alternative: compute full correlation matrix and extract cross-correlations
        combined_df = pd.concat([df[cols1], df[cols2]], axis=1)
        full_corr = combined_df.corr()
        # Extract the correlation between type1 and type2
        full_corr = full_corr.loc[cols1, cols2]
        results[(type1, type2)] = full_corr

  return results


df = pd.read_csv('NSCLC-Radiomics_features_full-original-fmcib+lifex+full-pyradiomics.csv')
correlations = pairwise_type_correlation(df)
# Identify high correlation pairs
high_corr_threshold = 0.5  # Adjust threshold as needed

for pair, corr in correlations.items():
    # print(f"\nAnalyzing {pair[0]} vs {pair[1]}:")
    # print(f"Correlation matrix shape: {corr.shape}")
    
    # # Find high correlations
    # high_corr_pairs = []
    
    # for i, row_name in enumerate(corr.index):
    #     for j, col_name in enumerate(corr.columns):
    #         corr_value = corr.iloc[i, j]
    #         if not pd.isna(corr_value) and abs(corr_value) >= high_corr_threshold:
    #             high_corr_pairs.append((row_name, col_name, corr_value))
    
    # print(f"Found {len(high_corr_pairs)} high correlation pairs (|r| >= {high_corr_threshold}):")
    
    # # Sort by absolute correlation value (descending)
    # high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    # for feature1, feature2, corr_val in high_corr_pairs:
    #     print(f"  {feature1} <-> {feature2}: r = {corr_val:.3f}")
    
    # # Save to CSV for further analysis
    # if high_corr_pairs:
    #     high_corr_df = pd.DataFrame(high_corr_pairs, 
    #                                columns=['Feature1', 'Feature2', 'Correlation'])
    #     high_corr_df.to_csv(f'high_correlations_{pair[0]}_{pair[1]}.csv', index=False)
        # print(f"  Saved to high_correlations_{pair[0]}_{pair[1]}.csv")

    # clustering = AgglomerativeClustering(linkage="complete", metric="precomputed", n_clusters = None, distance_threshold = 0).fit_predict(corr)
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap='coolwarm', vmin=-1, vmax=1, 
                xticklabels=False, yticklabels=False, 
                cbar_kws={'label': 'Correlation Coefficient'})
    
    plt.title(f'Correlation Heatmap: {pair[0]} vs {pair[1]}', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(f'nsclc_correlation_heatmap_{pair[0]}_{pair[1]}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    
    
    # ~/bhklab/radiomics/Projects/readii_2_roqc/data/results/{DATASET}/visualization/{feature_type}/{image_type}_heatmap.png