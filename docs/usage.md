# Usage Guide

## Project Configuration

TODO:: discuss how to edit the configuration files in the `config/` directory to match your research parameters

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
