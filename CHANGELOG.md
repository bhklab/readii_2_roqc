# Changelog

## 1.0.0 (2025-05-08)


### Features

* add conventional PR workflow ([44345f0](https://github.com/bhklab/readii_2_roqc/commit/44345f05ea4803452c57cf9d0b2a9684d25676af))
* add DMP directory READMEs ([e4f3f81](https://github.com/bhklab/readii_2_roqc/commit/e4f3f8179f5de59ce7b81bbdc98233f3d6b987c0))
* add function to combine all feature files - not working yet ([3cbc5e5](https://github.com/bhklab/readii_2_roqc/commit/3cbc5e5bf74613b28ba7e59bbcdf373d52e87fef))
* add gitignore so DMP is synced but none of the data inside ([b1ce26b](https://github.com/bhklab/readii_2_roqc/commit/b1ce26b81f170212b306a775a5c9c32b687b25f7))
* add handling for multiple masks per image ([370ab14](https://github.com/bhklab/readii_2_roqc/commit/370ab1400587fe86a35170e2b0ffd21ea6ba5d69))
* add handling in main for no NCs, return dataset proc dir from extract features to pass to combine_features ([1c138b8](https://github.com/bhklab/readii_2_roqc/commit/1c138b81f7f6e66b53fca8cd4e005f0b31fc03b6))
* add original image processing to feature extraction ([8add0b1](https://github.com/bhklab/readii_2_roqc/commit/8add0b172d028e9dc84d9d44beacb0a2d3fe7d0f))
* add pyradiomics extraction configurations for analysis ([bbc4939](https://github.com/bhklab/readii_2_roqc/commit/bbc493951dc51fb21466989ddbfd2a9da0b9c50c))
* add return of procdata dir from extract_features ([b7aac5e](https://github.com/bhklab/readii_2_roqc/commit/b7aac5e59dcfcf0efd4aafd7dd60ea48192a2db6))
* add single feature extraction config for testing purposes ([64d6b2d](https://github.com/bhklab/readii_2_roqc/commit/64d6b2d1fb7aaefcdef0163aaaa0ec250e030cb2))
* add workflows from template ([0f1f406](https://github.com/bhklab/readii_2_roqc/commit/0f1f406f88322bb46d750196f975096a8c25f113))
* added roi_name variable for directory path customization ([408a7b2](https://github.com/bhklab/readii_2_roqc/commit/408a7b2c39a128d05855a28496dec1682c84172c))
* h4h pyradiomics configuration ([553a68d](https://github.com/bhklab/readii_2_roqc/commit/553a68d32cd2c487bd14200da2701b677d5cc53e))
* ignore all contents of data directories ([6281a8c](https://github.com/bhklab/readii_2_roqc/commit/6281a8c18976e9d11efc189e32cdf780bde61fb3))
* implement PyRadiomic feature extraction pipeline ([3639180](https://github.com/bhklab/readii_2_roqc/commit/36391808dc7df38fdc503f301707dfa3705afeef))
* implement PyRadiomic feature extraction pipeline [#1](https://github.com/bhklab/readii_2_roqc/issues/1) ([3639180](https://github.com/bhklab/readii_2_roqc/commit/36391808dc7df38fdc503f301707dfa3705afeef))
* introduce image_types iterator to allow for feature extraction when negative controls are not passed to extract_features ([6d2a734](https://github.com/bhklab/readii_2_roqc/commit/6d2a734efa73c2436dfed1affc38173d2ec4575d))
* parallel negative control feature extraction working ([f07f4db](https://github.com/bhklab/readii_2_roqc/commit/f07f4db79b3d8bf8b823e5de0ccd68d6aeeeacb1))
* parallelizable pyradiomic feature extraction with readii negative controls ([e1bd8d9](https://github.com/bhklab/readii_2_roqc/commit/e1bd8d9731316e6fe9fe5c5684c078884d22bc26))
* run readii pipeline updates and data documentation start, add conventional PR workflow ([b8d7ac3](https://github.com/bhklab/readii_2_roqc/commit/b8d7ac347f04694093842f7876f1beb4f2bde86e))
* script to generate dataset index for pyradiomic feature extraction ([fe4ca85](https://github.com/bhklab/readii_2_roqc/commit/fe4ca8560f1c0767fad6068353e66e1532f416ca))
* script used to reorganize HN1 procdata for new ROI directory structure ([952885c](https://github.com/bhklab/readii_2_roqc/commit/952885cbbdb77d15ea5506d6b8cbad3ba8bd94c5))
* start readii run script for pyradiomic feature extraction ([a0d6743](https://github.com/bhklab/readii_2_roqc/commit/a0d6743e9bd3bf8384599d782d5c694524401513))


### Bug Fixes

* changed the sample level output files to be stored in ROI specific directories ([54233d1](https://github.com/bhklab/readii_2_roqc/commit/54233d15ed4d14ff2bf1d5ff4354917fa8f92517))
* correct default regions/transforms to be immutable ([255c8f7](https://github.com/bhklab/readii_2_roqc/commit/255c8f7e91a51db2274efbdd93bc35973d8cb84c))
* correct search for combine feature results to handle new proc ROI directory structure ([371db30](https://github.com/bhklab/readii_2_roqc/commit/371db305a41891c552f9d717bdbcc72291e0cf01))
* corrected variable to check for negative control manager existence in combine_feature_results ([ed7a389](https://github.com/bhklab/readii_2_roqc/commit/ed7a38960130ed301fb87972cae9aaa8967a609e))
