
# Developer Notes



#### [2025-05-20] Tracking extraction of all features for Aerts signature
From [Katy's notes](katy_notes.md):

|Feature / Dataset                        | HEAD-NECK-RADIOMICS-HN1 | NSCLC-Radiomics | RADCURE |
|-----------------------------------------|:-----------------------:|:---------------:|:-------:|
|original_firstorder_Energy               |             X           |        X        |    X    |
|original_shape_Compactness1              |             X           |        X        |    X    |
|original_glrlm_GrayLevelNonUniformity    |             X           |        X        |    X    |
|wavelet-HLH_glrlm_GrayLevelNonUniformity |          running        |   running       |    X    |


I think the paths in the run readii should be passed in a config file since I can't run two datasets at the same time.