
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

## LIFEx Guide
#### Documentation Links:
1. [User Guide](https://www.lifexsoft.org/images/phocagallery/documentation/LIFEx/UserGuide/LIFExUserGuide.pdf)
2. [Features](https://www.lifexsoft.org/images/phocagallery/documentation/LIFExFeatures/LIFExFeatures.pdf)
3. [Scripts](https://www.lifexsoft.org/images/phocagallery/documentation/LIFExScripts/LIFExScripts_v7.8.0.pdf)
4. [More](https://www.lifexsoft.org/index.php/resources/documentation)

#### Sample Script:
For the following feature extraction operation but on a batch of two series:

![lifex feature extraction UI](<Screenshot 2025-06-19 at 1.27.27 PM.png>)

```
## Lines with ## are comments

LIFEx.Output.Directory=PATH_TO_RESULTS_DIR

## _________________________________________________________________________________________________________________________

## [Patient0] section

LIFEx.Patient0.Series0=PATH_TO_SERIES_DIR
LIFEx.Patient0.Series0.Operation0=Texture,true,false,false,1,3d,Absolute,10.0,400.0,-1000.0,3000.0,1.0,1.0,1.0
LIFEx.Patient0.Roi0=PATH_TO_ROI_DIR

## _________________________________________________________________________________________________________________________

## [Patient1] section

LIFEx.Patient1.Series0=PATH_TO_SERIES_DIR
LIFEx.Patient1.Series0.Operation0=Texture,true,false,false,1,3d,Absolute,10.0,400.0,-1000.0,3000.0,1.0,1.0,1.0
LIFEx.Patient1.Roi0=PATH_TO_ROI_DIR

## _________________________________________________________________________________________________________________________

```

