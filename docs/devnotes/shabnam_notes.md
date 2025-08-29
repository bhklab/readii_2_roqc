
# Developer Notes



#### [2025-05-20] Tracking extraction of all features for Aerts signature
From [Katy's notes](katy_notes.md):

|Feature / Dataset                        | HEAD-NECK-RADIOMICS-HN1 | NSCLC-Radiomics | RADCURE |
|-----------------------------------------|:-----------------------:|:---------------:|:-------:|
|original_firstorder_Energy               |             X           |        X        |    X    |
|original_shape_Compactness1              |             X           |        X        |    X    |
|original_glrlm_GrayLevelNonUniformity    |             X           |        X        |    X    |
|wavelet-HLH_glrlm_GrayLevelNonUniformity |          running        |   running       |    X    |


#### [2025-05-27] FMCIB notebooks
NSCLC-Radiomics patient LUNG1-128 gtv lable is different.

#### [2025-06-02]
Correlation analysis is done on Aerts signature.

#### [2025-06-05]
fmcib pipline is now applicable on the up to date med-imgtools processed data.

#### [2025-06-10]
READII-FMCIB pipline is applied on NSCLC-Radiomics and HEAD-NECK-RADIOMICS-HN1 datasets.

#### [2025-06-18]
LIFEx feature are available for processed nifties from TCIA-NSCLC and TCIA-HN1 datasets.

#### [2025-08-12]
Image ids across different feature sources: 

fmcib:
image_path
procdata/TCIA_HEAD-NECK-RADIOMICS-HN1/images/cropped_images/cropped_bbox/original/HN1006_1.nii.gz

pyradiomics:
SampleID
HN1006_0001

lifex:
INFO_SeriesPath
/Users/shabnam/Downloads/HN1/HN1006_0001/CT_25815574/CT.nii.gz

#### [2025-08-22]
For calculating correlation between different sources:
I first made sure I have all the pyradiomics features in one file using scripts "shabnam/readii_2_roqc/workflow/scripts/analysis/python/merge_pyradiomics_features.py" 
Then I synced patients from different sources in one csv running "shabnam/readii_2_roqc/workflow/scripts/analysis/python/data_sync.py"
Then I calculated correlations and plotted them running "shabnam/readii_2_roqc/workflow/scripts/analysis/python/feature_corr_analysis.py"
