# Path to RadiomicSet RDS file to read in
filename <- ""

# Load in RadiomicSet RDS file to a MultiAssayExperiment object
mae <- readRDS(filename)

# Get the first assay, which contains features for one image type +  one feature extraction method
assay(mae, 1L) -> assay_bumpymatrix

# The dimensions of the assay will be (1, number of samples)
dim(assay_bumpymatrix)

# Dimension names will be the image modality and patient IDs
dimnames(assay_bumpymatrix)

# Get the features for the first sample, will be a CompressedSplitDFrameList
assay_bumpymatrix[1,1] -> sample_features

# The dimensions of the sample_features will be (1, number of features (including image info, diagnostics, and features))
dim(sample)