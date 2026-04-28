# Usage Guide

## Quick Start

1. Clone repository from <https://github.com/bhklab/readii_2_roqc>
1. Install environment with `pixi install`
1. Place image data in `data/rawdata/{YOUR_DATASET_NAME}/images`
1. Make dataset configuration file following `dataset_config_template.yaml` in `config/datasets`
1. Run med-imagetools with `pixi run mit {YOUR_DATASET_NAME} 'CT,RTSTRUCT' SEPARATE 'ROI:{YOUR_ROI_LIST}'`
1. Run readii with `pixi run readii_negative NSCLC-Radiomics false true 6`
1. Run feature extraction with `pixi run extract {YOUR_DATASET_NAME} pyradiomics linear_all_image_features.yaml`


## Environment
**IMPORTANT NOTE** This project will not work with Med-ImageTools properly with Python 3.12 or above. The pixi environment must have Python < 3.12 installed.

## Project Configuration

`config` should have three subdirectories: `datasets/`, `pyradiomics/`, and `signatures/`

### datasets
Each dataset needs a configuration file with the following settings filled in

```
DATA_SOURCE: ""    # where the data came from, will be used for data organization
DATASET_NAME: ""   # the name of the dataset , will be use for data organization

### CLINICAL VARIABLE INFORMATION ###
CLINICAL:
    FILE: ""                     # Name of the clinical data file associated with the data. Not a full path, just the name including the file suffix.
    OUTCOME_VARIABLES:
        event_label: ""          # Column name for survival event in the `FILE`, can be numeric, string, or bool
        event_value_mapping: {}  # Customize the `event_label` bool or string mapping to numeric type. Ex. {'Event value': 1, 'No event value': 0}
        time_label: ""           # Column name for survival time in the `FILE`, should be a numeric type
        convert_to_years: False  # Boolean, whether the `time_label` needs to be converted from days to years    
    EXCLUSION_VARIABLES: {}      # Column values of rows to drop in the clinical data (Ex. `{column_name: [val1, val2]}` )
    INCLUSION_VARIABLES: {}      # Column values of rows to select out of the clinical data

### MED-IMAGETOOLS settings
MIT:
    MODALITIES:                 # Modalities to process with autopipeline
        image: CT
        mask: RTSTRUCT     
    ROI_STRATEGY: MERGE         # How to handle multiple ROI matches 
    ROI_MATCH_MAP: KEY:ROI_NAME # Matching map for ROIs in dataset (use if you only want to process some of the masks in a segmentation)
                                # NOTE: there can be no spaces in KEY:ROI_NAME

### READII settings
READII:
    IMAGE_TYPES:                # Selection of image types to generate and perform feature extraction on (negative control settings)
        regions:                # Areas of image to apply permutation to
            - "full"
        permutations:           # Permutation type to apply to region
            - "shuffled"
        crop:
            - "cube"            # How to crop the image prior to feature extraction. Leave blank for no cropping.
        resize: [50]            # Size to crop the image to. Image will be resampled. Leave blank for no resampling.

### Feature Extraction settings
EXTRACTION:
    METHOD: pyradiomics                 # Extraction method to apply
    CONFIG: linear_all_images_features  # Configuration settings to use for extraction. Must have file in config/

### Analysis settings
ANALYSIS:   
    TRAIN_TEST_SPLIT:           # If using data for modelling, set up method of data splitting here
        split: False            # Whether to split the data
        split_variable: {}      # What variable from `CLINICAL.FILE` to use to split the data and values to group by (Ex. {'split_var': ['training', 'test']})
        impute: null            # What to impute values in `split_variable` with. Should be one of the values provided in `split_variable`. If none provided, won't impute, samples with no split value will be dropped.

RANDOM_SEED: 10                 # Seed for reproducibility of analysis.
```

### pyradiomics
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
|       |-- clinical
|       |   |-- {DATASET_NAME}_disease_subset.csv
|       |   `-- {DATASET_NAME}_outcome_data.csv
|       |-- correlations
|       |   `-- {extraction_method}
|       |       `-- {extraction_configuration_file_name}
|       |           |-- {image_type}_{correlation_method}_matrix.csv
|       |           `-- {image_type}_v_{image_type}_{correlation_method}_matrix.csv
|       |-- features
|       |   `-- {extraction_method}
|       |       |-- {crop}_{resize_x}_{resize_y}_{resize_z}
|       |       |   `-- {extraction_configuration_file_name}
|       |       |       `-- {PatientID}_{SampleNumber}
|       |       |           `-- {ROI_name}
|       |       |               |-- original_full_features.csv
|       |       |               |-- {permutation}_{region}_features.csv
|       |       |               `-- {permutation}_{region}_features.csv
|       |       `-- original_{size_x}_{size_y}_n 
|       |           |-- extraction_method_index.csv
|       |           `-- {extraction_configuration_file_name}
|       |               `-- {PatientID}_{SampleNumber}
|       |                   `-- {ROI_name}
|       |                       |-- original_full_features.csv
|       |                       |-- {permutation}_{region}_features.csv
|       |                       `-- {permutation}_{region}_features.csv
|       |-- images
|           |-- mit_{DATASET_NAME}
|           |   |-- {PatientID}_{SampleNumber}
|           |   |   |-- {ImageModality}_{SeriesInstanceUID}
|           |   |   |   `-- {ImageModality}.nii.gz
|           |   |   `-- {SegmentationModality}_{SeriesInstanceUID}
|           |   |       `-- {ROI_name}.nii.gz
|           |   |-- mit_{DATASET_NAME}_index-simple.csv
|           |   `-- mit_{DATASET_NAME}_index.csv
|           `-- readii_{DATASET_NAME}
|               |-- {crop}_{resize_x}_{resize_y}_{resize_z}
|               |   |-- {PatientID}_{SampleNumber}
|               |   |   `-- {ImageModality}_{SeriesInstanceUID}
|               |   |       |-- original_full.nii.gz    
|               |   |       |-- {permutation}_{region}.nii.gz
|               |   |       `-- {permutation}_{region}.nii.gz
|               |   `-- readii_{DATASET_NAME}_index.csv
|               `-- original_{size_x}_{size_y}_n
|                   |-- {PatientID}_{SampleNumber}
|                   |   `-- {ImageModality}_{SeriesInstanceUID}
|                   |       |-- {permutation}_{region}.nii.gz
|                   |       `-- {permutation}_{region}.nii.gz
|                   `-- readii_{DATASET_NAME}_index.csv
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
        |-- correlation
        |   `-- {extraction_method}
        |       |-- {extraction_configuration_file_name}
        |       `-- {signature_name}
        |-- features
        |   `-- {extraction_method}
        |       |-- {crop}_{resize_x}_{resize_y}_{resize_z}
        |       |   `-- {extraction_configuration_file_name}
        |       |       |-- original_full_features.csv
        |       |       `-- {permutation}_{region}_features.csv
        |       `-- original_{size_x}_{size_y}_n
        |           `-- {extraction_configuration_file_name}
        |               |-- original_full_features.csv
        |               `-- {permutation}_{region}_features.csv
        |-- prediction
        |   `-- {signature_name}
        |       |-- prediction_metrics.csv
        |       `-- hazards_{bootstrap_count}
        |           |-- original_full_features.csv
        |           |-- {permutation}_{region}_features.csv
        |           `-- {permutation}_{region}_features.csv
        `-- visualization
            |-- {signature_name} 
            `-- feature_correlation
                |-- cross
                |   `-- heatmap
                |       `-- {colormap}
                |           |-- {permutation}_{region}_{filter}_vs_{permutation}_{region}_{filter}_{correlation_method}_correlation_heatmap.png
                |           `-- {permutation}_{region}_{filter}_vs_original_full_{filter}_{correlation_method}_correlation_heatmap.png
                `-- self
                    `-- heatmap
                        `-- {colormap}
                            |-- original_full_{filter}_{correlation_method}_correlation_heatmap.png
                            `-- {permutation}_{region}_{filter}_{correlation_method}_correlation_heatmap.png
```


### Best Practices

- Store raw data in `data/rawdata/` and never modify it
- Store processed data in `data/procdata/` and all code used to generate it should be in `src/readii_2_roqc`
- Track data provenance (where data came from and how it was modified)
- Respect data usage agreements and licenses!
    This is especially important for data that should not be shared publicly



## Running Your Analysis
The pipeline is currently being run via `pixi` tasks. The following example shows how to run the pipeline using the `NSCLC-Radiomics` data.

# DICOM Image and Mask file processing with Med-ImageTools
This step converts the DICOM image files to NIfTI files, creates a unique ID for Image and Mask pairs, and generates an index file containing relevant metadata.

## Step 1: Run Med-ImageTools
This step converts the DICOM files to NIfTIs, assigns unique SampleIDs to image and mask pairs, and generates an index table for each file with associated metadata (e.g. DICOM tags)

```bash
pixi run mit NSCLC-Radiomics 'CT,RTSTRUCT' SEPARATE 'GTV:GTV-1,gtv-pre-op'
```

If you would like to customize the med-imagetools autopipeline run, you can run the following in the command line with changes:

```bash
imgtools autopipeline \
    --filename-format '{PatientID}_{SampleNumber}/{Modality}_{SeriesInstanceUID}/{ImageID}.nii.gz' \
    --modalities CT,RTSTRUCT \
    --roi-strategy SEPARATE \
    -rmap 'GTV:GTV-1,gtv-pre-op' \
    --spacing 1,1,1 \
    --window-level 8500 \
    --window-width 23000 \
    data/rawdata/TCIA_NSCLC-Radiomics/images \
    data/procdata/TCIA_NSCLC-Radiomics/images/mit_NSCLC-Radiomics
```

* Please keep the filename-format as shown for the remainder of the readii-2-roqc pipeline to complete as expected.
* The output procdata directory suffix after `mit_` must match the configuration file you created for this dataset.

## Step 2: Generate negative control images with READII
This step creates and saves READII negative controls specified in the config file for the provided dataset. 

It is recommended to run this in parallel for large datasets. Provided `n-4` jobs with n being the compute nodes available.

```bash
# Run the command with overwrite set to false, parallel set to true with 6 jobs
pixi run readii_negative NSCLC-Radiomics false true 6
```



### Step 3: Run feature extraction
This step first generates an index file for the specific feature extraction method, where each row contains the information for the image and mask pair to use.

```bash
pixi run extract NSCLC-Radiomics pyradiomics pyradiomics_original_all_features.yaml
```




## Survival Modelling
### Resources
https://scikit-survival.readthedocs.io/en/stable/user_guide/evaluating-survival-models.html#

Harrell's C-index: https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.metrics.concordance_index_censored.html
