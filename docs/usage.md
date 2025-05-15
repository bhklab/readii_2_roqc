# Usage Guide

## Project Configuration

`config` should have three subdirectories: `datasets/`, `extraction/`, and `signatures/`

### datasets
Each dataset needs a configuration file with the following settings filled in

```
`DATA_SOURCE` - where the data came from, will be used for data organization
`DATASET_NAME` - the name of the dataset , will be use for data organization

`MIT_INDEX_FILE` - name of the Med-ImageTools index file output by autopipeline when run on the image data.

`CLINICAL_FILE` - name of the clinical data file associated with the data. Not a full path, just the name including the file suffix.
`OUTCOME_VARIABLES`:
    `time_label`: "overall_survival_in_days" - column name for survival time in the `CLINICAL_FILE`
    `event_label`: "event_overall_survival" - column name for survival event in the `CLINICAL_FILE`
    `convert_to_years`: True - boolean, whether the `time_label` needs to be converted from days to years
    `event_value_mapping`: `{int: string | bool}` - if `event_label` values are not numeric, a dictionary can be provided to map the boolean or string to numbers. Ex: `{0: "Alive", 1: "Dead"}`

`EXCLUSION_VARIABLES`: `{column_name: [val1, val2]}` - column values of rows to drop in the clinical data 

`TRAIN_TEST_SPLIT`: 
    `split`: False - whether to apply a train test split to the data
    `split_variable`: `{split_label: ['train', 'test']}` - what column to use to split the data, and what values each of the subsets should have
    `impute`: null - value to impute any missing values in `split_variable` with so that all the data is categorized
```

### extraction
This directory should store any configuration settings used for feature extraction. They should be named/organized by the feature extraction method.

Example: `pyradiomics_original_all_features.yaml`

Different configuration set-ups can be documented here.

#### PyRadiomics

PyRadiomics feature extraction settings yaml files should be kept here. See the PyRadiomics ['Parameter File'](https://pyradiomics.readthedocs.io/en/latest/customization.html#parameter-file) documentation for details about this file.

### signatures

Files in this directory should list selected features in a radiomic signature and the corresponding weights from a fitted prediction model. 

!!! example "PyRadiomics CoxPH signature"
    ```lang-yaml 
    signature:
        'original_firstorder_Energy': 1.74e-11
        'original_shape_Compactness1': -1.65e+01
        'original_glrlm_GrayLevelNonUniformity': 4.95e-05
        'wavelet-HLH_glrlm_GrayLevelNonUniformity': 2.81e-06
    ```


---

## Data Setup

All data should be stored in a Data directory separate from this project directory. Within the project repo, there's a `data` directory containing `rawdata`, `procdata`, and `results` directories. The `rawdata` and `procdata` directories should by **symbolic links** pointing to the corresponding data directory in your separate Data directory.

### Aliasing `rawdata` and `procdata`

To set up the symbolic links for the `rawdata` and `procdata` directories, run the following commands, starting from your project directory:


```bash
ln -s /path/to/separate/data/dir/rawdata/{DiseaseRegion}/{DATASET_SOURCE}_{DATASET_NAME} data/rawdata/{DATASET_SOURCE}_{DATASET_NAME}

ln -s /path/to/separate/data/dir/procdata/{DiseaseRegion}/{DATASET_SOURCE}_{DATASET_NAME} data/procdata/{DATASET_SOURCE}_{DATASET_NAME}
```

!!! note
    You will need to perform this step for each dataset you wish to process with the READII-2-ROQC pipeline.

### Documenting datasets

When a new dataset has been added to the `rawdata` directory, you **MUST** document it on the [Data Sources](data_sources.md) page. 

Copy the following template and fill it in accordingly for each dataset. If anything about the dataset changes, make sure to keep this page up to date.

!!! example "Data Source Template"
    ```markdown
    NSCLC-Radiomics

    - **Name**: NSCLC-Radiomics (or Lung1)
    - **Version/Date**: Version 4: Updated 2020/10/22
    - **URL**: <https://www.cancerimagingarchive.net/collection/nsclc-radiomics/>
    - **Access Method**: NBIA Data Retriever
    - **Access Date**: 2025-04-23
    - **Data Format**: DICOM
    - **Citation**: Aerts, H. J. W. L., Wee, L., Rios Velazquez, E., Leijenaar, R. T. H., Parmar, C., Grossmann, P., Carvalho, S., Bussink, J., Monshouwer, R., Haibe-Kains, B., Rietveld, D., Hoebers, F., Rietbergen, M. M., Leemans, C. R., Dekker, A., Quackenbush, J., Gillies, R. J., Lambin, P. (2014). Data From NSCLC-Radiomics (version 4) [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2015.PF0M9REI 
    - **License**: [CC BY-NC 3.0](https://creativecommons.org/licenses/by-nc/3.0/)
    - **Data Types**: 
        - Images: CT, RTSTRUCT
        - Clinical: CSV
    - **Sample Size**: 422 subjects
    - **ROI Name**: Tumour = GTV-1
    - **Notes**: LUNG-128 does not have a GTV segmentation, so only 421 patients are processed.
    ```


### Project `data` Directory Tree

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


### Best Practices

- Store raw data in `data/rawdata/` and never modify it
- Store processed data in `data/procdata/` and all code used to generate it should be in `workflow/scripts/`
- Track data provenance (where data came from and how it was modified)
- Respect data usage agreements and licenses!
    This is especially important for data that should not be shared publicly



## Running Your Analysis



## Data splitting
### aerts_original
Train: NSCLC-Radiomics  
Validation: HN1, RADCURE-test

### aerts_RADCURE_refit
Train: RADCURE-train  
Validation: HN1, RADCURE-test

### r2r_NSCLC
Train: NSCLC-Radiomics  
Validation: HN1, RADCURE-test

### r2r_RADCURE
Train: RADCURE-train  
Validation: HN1, RADCURE-test


## Survival Modelling
### Resources
https://scikit-survival.readthedocs.io/en/stable/user_guide/evaluating-survival-models.html#

Harrell's C-index: https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.metrics.concordance_index_censored.html
