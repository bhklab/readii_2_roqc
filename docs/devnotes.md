# Developer Notes

## Data Processing Notes
[2025-05-05] PyRadiomics original_all_features extraction tracking
[2025-05-07] Updated with completed HN1 runs

|NC / Dataset        | CC-Radiomics-Phantom | HEAD-NECK-RADIOMICS-HN1 | NSCLC-Radiomics |
|--------------------|:--------------------:|:-----------------------:|:---------------:|
|full original       |           X          |           X             |        X        |
|full randomized     |           X          |           X             |        X        |
|full sampled        |           X          |           X             |        X        |
|full shuffled       |           X          |           X             |        X        |
|non_roi randomized  |           X          |           X             |        X        |
|non_roi sampled     |           X          |           X             |        X        |
|non_roi shuffled    |           X          |           X             |        X        |
|roi randomized      |           X          |           X             |        X        |
|roi sampled         |           X          |           X             |        X        |
|roi shuffled        |           X          |           X             |        X        |


[2025-05-12] Tracking extraction of all features for Aerts signature
|Feature / Dataset                        | HEAD-NECK-RADIOMICS-HN1 | NSCLC-Radiomics | RADCURE |
|original_firstorder_Energy               |             X           |        X        |    X    |
|original_shape_Compactness1              |             X           |        X        |    X    |
|original_glrlm_GrayLevelNonUniformity    |             X           |        X        |    X    |
|wavelet-HLH_glrlm_GrayLevelNonUniformity |          running        |   running       |    X    |

* Bootstrapping help came from: https://acclab.github.io/bootstrap-confidence-intervals.html
* survcomp R package only works for linux and osx-64
* tried the scikit-survival implementation of the concordance index with bootstrapping, but results don't match Mattea's exactly
* trying with R now

[2025-05-13] Processing data for signature prediction models
* Since RADCURE was processed with old med-imagetools, manually set up mit_index from dataset.csv
    * Change the column `patient_ID` to `PatientID`
    * Add index column label `SampleID`

|File / Dataset       | HEAD-NECK-RADIOMICS-HN1 | NSCLC-Radiomics | RADCURE             |
| clinical            | id                      | PatientID       | patient_id          |
| mit index file      | SampleID, PatientID     | PatientID       | SampleID, PatientID |
| radiomics           | ID                      | ID              | ID                  |