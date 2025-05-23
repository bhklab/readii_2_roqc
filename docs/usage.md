# Usage Guide

## Project Configuration

`config` should have three subdirectories: `datasets/`, `extraction/`, and `signatures/`

### datasets
Each dataset needs a configuration file with the following settings filled in

```
DATA_SOURCE: ""    # where the data came from, will be used for data organization
DATASET_NAME: ""   # the name of the dataset , will be use for data organization

### CLINICAL VARIABLE INFORMATION ###
CLINICAL:
    FILE: ""                     # Name of the clinical data file associated with the data. Not a full path, just the name including the file suffix.
    OUTCOME_VARIABLES:
        time_label: ""           # Column name for survival time in the `FILE`, should be a numeric type
        event_label: ""          # Column name for survival event in the `FILE`, can be numeric, string, or bool
        convert_to_years: False  # Boolean, whether the `time_label` needs to be converted from days to years
        event_value_mapping: {}  # Customize the `event_label` bool or string mapping to numeric type. Should be in the order {0: Alive_value, 1: Dead_value}
    EXCLUSION_VARIABLES: {}      # Column values of rows to drop in the clinical data (Ex. `{column_name: [val1, val2]}` )

### MED-IMAGETOOLS settings
MIT:
    MODALITIES: CT,RTSTRUCT     # Modalities to process with autopipeline
    ROI_STRATEGY: MERGE         # How to handle multiple ROI matches 
    ROI_MATCH_MAP:              # Matching map for ROIs in dataset (use if you only want to process some of the masks in a segmentation)
        KEY:ROI_NAME            # NOTE: there can be no spaces in KEY:ROI_NAME

### READII settings
READII:
    IMAGE_TYPES:                # Selection of image types to generate and perform feature extraction on (negative control settings)
        regions:                # Areas of image to apply permutation to
            - "full"
        permutations:           # Permutation type to apply to region
            - "original"
        crop:                   # How to crop the image prior to feature extraction
    TRAIN_TEST_SPLIT:           # If using data for modelling, set up method of data splitting here
        split: False            # Whether to split the data
        split_variable: {}      # What variable from `CLINICAL.FILE` to use to split the data and values to group by (Ex. {'split_var': ['training', 'test']})
        impute: null            # What to impute values in `split_variable` with. Should be one of the values provided in `split_variable`. If none provided, won't impute, samples with no split value will be dropped.

RANDOM_SEED: 10                 # Seed for reproducibility of analysis.
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
|       |       |-- extraction_method_index.csv
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
