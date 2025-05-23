# Developer Notes

## Data Processing Notes
#### [2025-05-05] PyRadiomics original_all_features extraction tracking
#### [2025-05-07] Updated with completed HN1 runs

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


#### [2025-05-12] Tracking extraction of all features for Aerts signature
|Feature / Dataset                        | HEAD-NECK-RADIOMICS-HN1 | NSCLC-Radiomics | RADCURE |
|original_firstorder_Energy               |             X           |        X        |    X    |
|original_shape_Compactness1              |             X           |        X        |    X    |
|original_glrlm_GrayLevelNonUniformity    |             X           |        X        |    X    |
|wavelet-HLH_glrlm_GrayLevelNonUniformity |          running        |   running       |    X    |

* Bootstrapping help came from: https://acclab.github.io/bootstrap-confidence-intervals.html
* survcomp R package only works for linux and osx-64
* tried the scikit-survival implementation of the concordance index with bootstrapping, but results don't match Mattea's exactly
* trying with R now

#### [2025-05-13] Processing data for signature prediction models
* Since RADCURE was processed with old med-imagetools, manually set up mit_index from dataset.csv
    * Change the column `patient_ID` to `PatientID`
    * Add index column label `SampleID`

|File / Dataset       | HEAD-NECK-RADIOMICS-HN1 | NSCLC-Radiomics | RADCURE             |
| clinical            | id                      | PatientID       | patient_id          |
| mit index file      | SampleID, PatientID     | PatientID       | SampleID, PatientID |
| radiomics           | ID ==                      | ID              | ID                  |


* HN1 - going to make a SampleID column in the mit2_index 
* Rerunning all the MIT and feature extraction


## Manuscript Notes
#### [2025-05-14] Manuscript Review Feedback Notes

**Introduction/Background**

- [ ] Paragraph around biological/clinical application of radiomics (why do we do it?)
- [ ] Literature review for other quality control methods for radiomics

**Methods**

- [ ] Add prediction of HPV status model
- [ ] Include section about saving out the negative control images
- [ ] Explain modular implementation of the negative controls such that users can construct their own
- [ ] Add ability to save out the changed mask from the contraction/expansion NCs
- [X] Look for synonyms for transformation
    * Intensity transformation? --> permutation?
- [ ] Sisira asked for an example of the RadiomicSet data to understand what it looks like


**Results**

- [X] Josh liked the new abstract figure more
- [ ] Get p-value calculation code from Caryn
- [ ] Save out bootstrap hazards so they don't get recalculated
- [ ] Keep the diagonal self-correlation plots, will go in supplemental
- [ ] Calculate and plot average correlations between clusters of features (shape vs. first order)
- [ ] Hierarchical clustering between the feature class clusters
- [ ] Correlation plot of the Aerts signature + volume features
- [ ] Compare distribution of correlation values in a line plot
    * Line for each image type
    * x-axis is the feature types, colourblock behind the plot for each imagetype
- [ ] Plot correlations with outcome before and after QC
 
*Plots*

- [ ] Box plot of hazards for each image type and dataset to compare


## Project Repo Organization Notes

#### [2025-05-14] data and workflow organization

```bash
data
|-- procdata
|   `-- {DATASET_SOURCE}_{DATASET_NAME} --> /path/to/separate/data/dir/procdata/{DiseaseRegion}/{DATASET_SOURCE}_{DATASET_NAME}
|       |-- correlations
|       |-- features
|       |   `-- {extraction_method}
|       |       `-- {extraction_configuration_file_name}
|       |           `-- {PatientID}_{SampleNumber}
|       |               `-- {ROI_name}
|       |                   |-- full_original_features.csv
|       |                   |-- {neg_control_region}_{neg_control_permutation}_features.csv
|       |                   `-- {neg_control_region}_{neg_control_permutation}_features.csv
|       |-- images
|       |   |-- mit_{DATASET_NAME}
|       |   |   `-- {PatientID}_{SampleNumber}
|       |   |       |-- {ImageModality}_{SeriesInstanceUID}
|       |   |       |   `-- {ImageModality}.nii.gz
|       |   |       `-- {SegmentationModality}_{SeriesInstanceUID}
|       |   |           `-- {ROI_name}.nii.gz
|       |   `-- readii_{DATASET_NAME}
|       |       `-- {PatientID}_{SampleNumber}
|       |           `-- {ImageModality}_{SeriesInstanceUID}
|       |               |-- {neg_control_region}_{neg_control_permutation}.nii.gz
|       |               `-- {neg_control_region}_{neg_control_permutation}.nii.gz
|       `-- signatures
|           `-- {signature_name}
|               |-- full_original_signature_features.csv
|               `-- {neg_control_region}_{neg_control_permutation}_signature_features.csv
|-- rawdata
|   `-- {DATASET_SOURCE}_{DATASET_NAME} --> /path/to/separate/data/dir/srcdata/{DiseaseRegion}/{DATASET_SOURCE}_{DATASET_NAME}
|       |-- clinical
|       |   `-- {Clinical Data File}.csv OR {Clinical Data File}.xlsx
|       `-- images
|           `-- {DATASET_NAME}
|               |-- {Sample1 DICOM directory}
|               |-- {Sample2 DICOM directory}
|               |-- ...
|               `-- {SampleN DICOM directory}
`-- results
    `-- {DATASET_SOURCE}_{DATASET_NAME}
        |-- correlation_figures
        |-- features
        |   `-- {extraction_method}
        |       `-- {extraction_configuration_file_name}        
        `-- signature_performance
            `-- {signature_name}.csv
                |-- full_original_features.csv
                |-- {neg_control_region}_{neg_control_permutation}_features.csv
                `-- {neg_control_region}_{neg_control_permutation}_features.csv
```

```bash
workflow
|-- notebooks
`-- scripts
    |-- analysis
    |   |-- python
    |   |   `-- predictive_signature_testing.py
    |   `-- r
    |       |-- io.r
    |       `-- survival_prediction.r
    |-- feature_extraction
    |   `-- pyradiomics_index.py
    |-- orcestra
    |   `-- read_radiomicset.r
    |-- mit
    |   `-- run_autopipeline.sh
    `-- readii
        `-- run_readii.py
```

#### [2025-05-22] MIT Snakemake and config updates
* wrote Snakemake script to run autopipeline for the dataset
* Updated the datasets/config file format and updated usage documentation.


#### [2025=05-23] MIT Snakemake reorg
* Made smk file just for MIT rules, made run_MIT rule in main Snakefile