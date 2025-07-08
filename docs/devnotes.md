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


#### [2025=05-23] MIT Snakemake reorg + pyradiomics_index refactor
* Made smk file just for MIT rules, made run_MIT rule in main Snakefile

`pyradiomics_index --> index`

* Added click CLI input
* Made a genral index function that calls the pyradiomics index function
* Want to add a dataset config variable or CLI argument that sets the method of feature extraction


#### [2025-05-26] Index and run READII reorg, start of correlation coding
`katys/refactor-pyradiomics-index` - has updated code for generating the index file for PyRadiomics using click for CLI

`katys/refactor-run_readii` - has new file called `make_negative_controls.py` that just generates and saves the negative control images, no feature extraction
* uses click
* has function to get just the Image Type settings from the READII section of the config

`katys/add-correlation-calculation` - copied the `run_analysis.ipynb` notebook from `readii-fmcib`
* Started updating the config settings at the top of the file to hopefully run the correlation analysis and start generating figures

All of these are waiting on readii 1.36.2 to be able to install from PyPI to work


#### [2025-05-27] 
* Debugging the overwrite issue with make_negative_controls
* Solved by using Series to get image metadata
* Also need to run alignImages whenever flattenImage is run so that origin, direction, and **spacing** are maintained -- made this an issue in READII as well


#### [2025-05-28]
* Added make_negative_controls to Snakefile in run_readii rule
* Need to figure out how to list all the output files 
* From Jermiah:
    * https://snakemake.readthedocs.io/en/stable/snakefiles/rules.html#data-dependent-conditional-execution 
    * you would make the autopipeline rule a checkpoint then you can create an input function for another rule that parses the index file to generate all the files you want
* Trying to run it as is with full NSCLC-Radiomics dataset
* Plan for feature extraction refactor:
    * Will generate an index for each image type - ID, Image, Mask
    * Make Snakemake rule run per image type
    * Feature extraction script will parallelize by patient 

#### [2025-05-29]
* Need to add MRI handling to make_negative_controls
* Changing how config MODALITIES is set up so that image and mask are separate


#### [2025-06-04]
* Actually need the READII index earlier, for the negative control creation even
* Essentially making the edges file MIT used to make
* Need to generate ID for unique image-mask pair 
    * OR process all masks with a single image. The full would be the same for each of them
    * Will end up with duplicates of the full mask for each mask - unless I rearrange the outputs

```bash
data
|-- procdata
|   `-- {DATASET_SOURCE}_{DATASET_NAME} --> /path/to/separate/data/dir/procdata/{DiseaseRegion}/{DATASET_SOURCE}_{DATASET_NAME}
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
|       |               |-- {SegmentationModality}_{SeriesInstanceUID}
|       |               |   |-- {neg_control_permutation}_roi_.nii.gz
|       |               |   `-- {neg_control_permutation}_non_roi_.nii.gz 
|       |               |-- full_{neg_control_permutation}.nii.gz
|       |               `-- full_{neg_control_permutation}.nii.gz
```
This is what I want to end up with eventually, but for now am going to leave the full directories inside the segmentation file

#### [2025-06-09]
* Could also see about making the roi region name the actual roi name
    * That might make processing difficult though, since every ROI negative control image will have a different name

#### [2025-06-10]
* So the NIFTIWriter has an index saver portion of it from Med-ImageTools
* Using that during negative control generation
* Just need to add the Mask paths to it


#### [2025-06-16]
* Figured out how to use the NIFTIWriter the way I want
* Updated the values used for saving the negative control NIFTI's so index file has columns I can use like Med-ImageTools
* Should talk to Jermiah about making original image a READII filetype
* Updated generate_pyradiomics_index
    * Currently set up to expect the original image index from med-imagetools always and can pass the readii index if available


#### [2025-07-08]
* In sample_feature_writer, use a semi-colon to separate the keys and values of the feature vector so it can be read in easier by pandas
    * With the comma, the pyradiomic settings lines confuse it.
* Could also transpose this and have the keys on line 1 and vals on line 2, not sure it would solve the problem