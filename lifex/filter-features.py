import pandas as pd

# Path to lifex-output CSV file
df = pd.read_csv('PATH')

columns_to_keep = [

    'INFO_SeriesPath',

    # Conventional Indices
    'INTENSITY-BASED_MinimumIntensity(IBSI:1GSF)[]',   # Min Value
    'INTENSITY-BASED_MeanIntensity(IBSI:Q4LE)[]',      # Mean Value
    'INTENSITY-BASED_StandardDeviation(IBSI:No)[]',    # Std Value
    'INTENSITY-BASED_MaximumIntensity(IBSI:84IY)[]',   # Max Value

    # Histogram
    'INTENSITY-HISTOGRAM_IntensityHistogramSkewness(IBSI:88K1)[Intensity]',  # Histogram Skewness
    'INTENSITY-HISTOGRAM_IntensityHistogramKurtosis(IBSI:C3I7)[Intensity]',  # Kurtosis
    'INTENSITY-HISTOGRAM_IntensityHistogramEntropyLog2(IBSI:TLU2)[Intensity]', # Entropy
    'INTENSITY-BASED_IntensityBasedEnergy(IBSI:N8CA)[]',  # Energy

    # Shape
    'MORPHOLOGICAL_Volume(IBSI:RNU0)[cm3]',  # Shape Volume (ml)

    # GLCM (Gray-Level Co-occurrence Matrix)
    'GLCM_AngularSecondMoment(IBSI:8ZQL)',   # Homogeneity (ASM)
    'INTENSITY-BASED_IntensityBasedEnergy(IBSI:N8CA)[]',  # Energy (already above)
    'GLCM_Contrast(IBSI:ACUI)',              # Contrast
    'GLCM_Correlation(IBSI:NI2N)',           # Correlation
    'GLCM_JointEntropyLog2(IBSI:TU9B)',      # Entropy (GLCM)
    'GLCM_Dissimilarity(IBSI:8S9J)',         # Dissimilarity

    # GLRLM (Gray-Level Run Length Matrix)
    'GLRLM_ShortRunsEmphasis(IBSI:22OV)',    # SRE
    'GLRLM_LongRunsEmphasis(IBSI:W4KF)',     # LRE
    'GLRLM_LowGreyLevelRunEmphasis(IBSI:V3SW)',  # LGRE
    'GLRLM_HighGreyLevelRunEmphasis(IBSI:G3QZ)', # HGRE
    'GLRLM_ShortRunLowGreyLevelEmphasis(IBSI:HTZT)', # SRLGE
    'GLRLM_ShortRunHighGreyLevelEmphasis(IBSI:GD3A)', # SRHGE
    'GLRLM_LongRunLowGreyLevelEmphasis(IBSI:IVPO)',   # LRLGE
    'GLRLM_LongRunHighGreyLevelEmphasis(IBSI:3KUM)',  # LRHGE
    'GLRLM_GreyLevelNonUniformity(IBSI:R5YN)',        # GLNUr
    'GLRLM_RunLengthNonUniformity(IBSI:W92Y)',        # RLNU
    'GLRLM_RunPercentage(IBSI:9ZK5)',                 # RP

    # NGTDM
    'NGTDM_Coarseness(IBSI:QCDE)',                    # Coarseness
    'NGTDM_Contrast(IBSI:65HE)',                      # ContrastN

    # GLSZM (Gray-Level Zone Length Matrix)
    'GLSZM_SmallZoneEmphasis(IBSI:5QRC)',             # SZE
    'GLSZM_LargeZoneEmphasis(IBSI:48P8)',             # LZE
    'GLSZM_LowGrayLevelZoneEmphasis(IBSI:XMSY)',      # LGZE
    'GLSZM_HighGrayLevelZoneEmphasis(IBSI:5GN9)',     # HGZE
    'GLSZM_SmallZoneLowGreyLevelEmphasis(IBSI:5RAI)', # SZLGE
    'GLSZM_SmallZoneHighGreyLevelEmphasis(IBSI:HW1V)',# SZHGE
    'GLSZM_LargeZoneLowGreyLevelEmphasis(IBSI:YH51)', # LZLGE
    'GLSZM_LargeZoneHighGreyLevelEmphasis(IBSI:J17V)',# LZHGE
    'GLSZM_GreyLevelNonUniformity(IBSI:JNSA)',        # GLNUz
    'GLSZM_ZoneSizeNonUniformity(IBSI:4JP3)',         # ZLNU
    'GLSZM_ZonePercentage(IBSI:P30P)',                # ZP
]

# Filter the DataFrame
filtered_df = df[columns_to_keep]

# Save the filtered DataFrame to a new CSV file
filtered_df.to_csv('PATH', index=False)
