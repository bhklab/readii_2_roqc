# Usage Guide

## Project Configuration

`config` should have two subdirectories: `datasets/` and `pyradiomics/`

### `datasets`

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

## `pyradiomics`

PyRadiomics feature extraction settings yaml files should be kept here. See the PyRadiomics ['Parameter File'](https://pyradiomics.readthedocs.io/en/latest/customization.html#parameter-file) documentation for details about this file.

## Data Setup

TODO:: discuss how to add your input data to the `data/rawdata/` directory and document it properly in the `docs/data_sources/` directory

TODO:: discuss how to manage and organize your data sources effectively

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
