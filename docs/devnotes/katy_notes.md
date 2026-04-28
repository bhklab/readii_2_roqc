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
- [X] Save out bootstrap c-indices so they don't get recalculated
- [ ] Keep the diagonal self-correlation plots, will go in supplemental
- [ ] Calculate and plot average correlations between clusters of features (shape vs. first order)
- [ ] Hierarchical clustering between the feature class clusters
- [ ] Correlation plot of the Aerts signature + volume features
- [ ] Compare distribution of correlation values in a line plot
    * Line for each image type
    * x-axis is the feature types, colourblock behind the plot for each imagetype
- [ ] Plot correlations with outcome before and after QC
 
*Plots*

- [X] Box plot of hazards (c-indices) for each image type and dataset to compare


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

#### [2025-07-09]
* Existing code already performed the transpose, just needed to change the comma to a semi-colon and fix the sortby value
* Extraction is working! Adding a tqdm progress bar then running CPTAC-CCRCC

#### [2025-07-17]
* Merged rearranged workflow into main branch
* Pipeline can now be run with pixi tasks
* Snakemake needs to be updated and tested

#### [2025-07-21]
* For NSCLC-Radiogenomics, R01-003 SEG file can be loaded by pydicom, but highdicom fails with an IntegrityError:

```python
image = dcmread("TCIA_NSCLC-Radiogenomics/images/NSCLC-Radiogenomics/R01-003/12-09-1991-NA-CT_CHEST_WO-02622/1000.000000-3D_Slicer_segmentation_result-96510/1-1.dcm")
seg = hd.seg.Segmentation.from_dataset(image)

## ERROR MESSAGE
File ~/Documents/BHKLab_GitHub/readii_2_roqc/.pixi/envs/default/lib/python3.12/site-packages/highdicom/image.py:4185, in _Image._create_ref_instance_table(self, referenced_uids)
   4177 with self._db_con:
   4178     self._db_con.execute(
   4179         "CREATE TABLE InstanceUIDs("
   4180         "StudyInstanceUID VARCHAR NOT NULL, "
   (...)   4183         ")"
   4184     )
-> 4185     self._db_con.executemany(
   4186         "INSERT INTO InstanceUIDs "
   4187         "(StudyInstanceUID, SeriesInstanceUID, SOPInstanceUID) "
   4188         "VALUES(?, ?, ?)",
   4189         referenced_uids,
   4190     )

IntegrityError: UNIQUE constraint failed: InstanceUIDs.SOPInstanceUID
```

#### [2025-07-23]
* Got a followup on the above error: <https://github.com/ImagingDataCommons/highdicom/issues/370>

* Added prediction functionality to the pipeline
* Train or test subsetting is specified when running pixi run predict - it will run the prediction on that subset of patients
    * Will have to run once for train and once for test
    * With the way I currently have it programmed, this is the best setup
    * In a refactor, could make it so that train and test is run together and split, but it was too complicated and would duplicate too much code with the current setup

* Shabnam found a couple of bugs:
    1. If a MaskID has any spaces in it, the extractor breaks
    2. If one READII region is already extracted, need to specify overwrite to get any additional ones extracted

#### [2025-07-31]
* For docker containerization to use on H4H, need to run the following
```bash
docker build --platform linux/amd64-r2r -t bhklabkscott/amd64-r2r
```

* Can then push to docker hub
```bash
docker push bhklabkscott/amd64-r2r
```

* On H4H, in the readii_2_roqc git directory run
```bash
apptainer pull docker://bhklabkscott/amd64-r2r

apptainer shell amd64-r2r_latest.sif
```

## OPC Prediction 
#### [2025-08-18]  
* Reading the Aerts et al., Sun et al., Reiazi et al. papers to get methodology for OPC prediction
* Discovery dataset: RADCURE
* Validation dataset: HN1 (HNSCC maybe)

Sample Size  
* confirmed primary tumour oropharynx
* underwent treatment with curative intent
* HPV test available

Methodology  
* Aerts: pick most prognostic feature from each group (shape, first order, texture, wavelet)
* Sun: linear elastic model for feature selection and model building  
    * lambda defined by cross-validation, 0.2
    * alpha set to 0.5 after grid search
* Reiazi: mRMRe feature selection to 1000 features, imbalance adjustment by undersampling majority class, random forest classifier to predict HPV status, gridsearch for # of trees, max depth, min num samples at leaf, 5-fold cross-validation, train-test 100 times, final score based on average prediction score (1000 times)
* Choi et al.: Boruta feature selection, generalized linear models to plot ROC curves

dev_binary_prediction.ipynb  
* predict code is all for continuous and relies on mit index
* writing different code for OPC prediction that does binary prediction and uses the feature extraction index
* Looks like some of the OPC patients don't have GTV feature data
* need to filter clinical data for this
* then can do the feature stuff

#### [2025-08-19]
* RADCURE: 418 OPC tumor samples with HPV status
* HN1: 80 OPC tumor samples with HPV status
* HNSCC: 292 OPC tumor samples with HPC

* scikit-learn  
    * Can't install latest version (1.7.1) because scikit-survival==0.24.1 depends on scikit-learn>=1.6.1,<1.7 

* pycaret
    * had to make a separate environment to handle numpy mismatch in r2r and pycaret
    * models environment has pycaret


#### [2025-08-20]
* Tried using glmnet like in the Choi 2020 paper
    * Tried in Python:  
        * https://pypi.org/project/glmnet/ - broken, won't install, archived
        * https://github.com/bbalasub1/glmnet_python - also broken on install
    * Tried in R:
        * https://glmnet.stanford.edu/articles/glmnet.html
        * Got this running in r2r_glmnet on lab server
        * Not enough details described in the paper to recreate the analysis
        * No idea how to get the prediction and AUC
    * Predicting with pycaret instead just using the subset of features
        * On RADCURE, got an AUC of 0.6315

* Trying survival prediction for OPC patients with Choi's survival signature

* Had to update predict.py to use the feature extraction index generated by r2r - this makes more sense than using the mit index anyway
* READII has some bugs in it for converting string outcome variables to ints - do the conversion in predict.py for now until it's fixed

* Leaving HEAD-NECK-RADIOMICS-HN1 to rerun feature extraction on screen - HN1-rerun

* Trying to fix the filtering for HNSCC for making the negative controls - running into issues with the multiple CTs but can't figure out what to do


## Correlations

#### [2025-09-10]
* Got self-correlation plots made for the original features of each PyRadiomics feature class for NSCLC-Radiomics
* Should be able to set up loop to process all the HN1 original features tomorrow
* And hopefully RADCURE negative controls will be done generating by then, and can run feature extraction



## Adding FMCIB
#### [2025-09-23]

* Updated the index script to generate the FMCIB extraction file expected
* Realized I need to update make_negative_controls to add in the crop method
* Can use this as an opportunity to parallelize the script at the same time
* Restructuring in a notebook first


## Debugging updated negative control generation
#### [2025-10-14]

* RADCURE data experiences overflow errors for the randomized full negative control feature extraction

```python
/readii_2_roqc/.pixi/envs/default/lib/python3.11/site-packages/radiomics/imageoperations.py:127: RuntimeWarning: overflow encountered in scalar subtract
  lowBound = minimum - (minimum % binWidth)
/readii_2_roqc/.pixi/envs/default/lib/python3.11/site-packages/radiomics/imageoperations.py:132: RuntimeWarning: overflow encountered in scalar add
  highBound = maximum + 2 * binWidth
/readii_2_roqc/.pixi/envs/default/lib/python3.11/site-packages/radiomics/imageoperations.py:134: RuntimeWarning: overflow encountered in scalar subtract
  binEdges = numpy.arange(lowBound, highBound, binWidth)
/readii_2_roqc/.pixi/envs/default/lib/python3.11/site-packages/radiomics/imageoperations.py:134: RuntimeWarning: overflow encountered in scalar add
  binEdges = numpy.arange(lowBound, highBound, binWidth)
```

* So digging into this revealed it's an issue when the range of voxel values in an image exceeds what can be handled by np.int64
* Trying out windowing the image to see if this solves the problem
    * Running med-imagetools autopipeline with the following

    ```bash
    imgtools autopipeline \
    --filename-format '{PatientID}_{SampleNumber}/{Modality}_{SeriesInstanceUID}/{ImageID}.nii.gz' \
    --modalities CT,RTSTRUCT \
    --roi-strategy SEPARATE \
    -rmap "ROI:GTVp" \
    --window-level 8500 \ # important change
    --window-width 23000 \ # important change
    data/rawdata/TCIA_RADCURE_test/images \
    data/procdata/TCIA_RADCURE_window_test/images/mit_RADCURE_window_test
    ```

* OHHH I think this is the original randomized bug!! Why the randomized wouldn't run - that's why I haven't run into until now.

* Ok SO in order to handle this bug there are two updated settings to be applied in the full pipeline

1. In autopipeline, use windowing settings:
    ```
    --window-level 1500 
    --window-width 7000 
    ```
    This will set the range of values to -2000 to 5000 in the processed nifti images. This range was chosen to maintain the artifacts in the RADCURE dataset while preventing the overflow errors in the GLCM calculations during PyRadiomics feature extraction on the randomized negative control images.

2. In feature extraction, set the interpolator to `sitk.Linear`. This is better for CT images and prevents the interpolated values from falling outside the -2000 to 5000 range we made in the windowing of med-imagetools.

Now need to rerun all of the datasets with these updated settings. The former shouldn't have an impact on the other two datasets (HN1 and Lung1), but the latter will because we're impacting the interpolation during feature extraction.