# Data Sources

## CC-Radiomics-Phantom
- **Name**: Credence Cartridge Radiomics Phantom CT Scans
- **Version/Date**: Version 1: Updated 2017/07/28
- **URL**: <https://www.cancerimagingarchive.net/collection/cc-radiomics-phantom/>
- **Access Method**: NBIA Data Retriever
- **Access Date**: 2025-04-22
- **Data Format**: DICOM
- **Citation**: Mackin, D., Ray, X., Zhang, L., Fried, D., Yang, J., Taylor, B., Rodriguez-Rivera, E., Dodge, C., Jones, A., & Court, L. (2017). Data From Credence Cartridge Radiomics Phantom CT Scans (CC-Radiomics-Phantom) [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2017.zuzrml5b 
- **License**: [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/)
- **Data Types**: 
    - Images: CT, RTSTRUCT
- **Sample Size**: 17 subjects


## NSCLC-Radiomics
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
- **Notes**: 


## HEAD-NECK-RADIOMICS-HN1
- **Name**: HEAD-NECK-RADIOMICS-HN1 (or H&N1)
- **Version/Date**: Version 3: Updated 2020/07/29
- **URL**: <https://www.cancerimagingarchive.net/collection/head-neck-radiomics-hn1/>
- **Access Method**: NBIA Data Retriever
- **Access Date**: 2025-04-23
- **Data Format**: DICOM
- **Citation**: Wee, L., & Dekker, A. (2019). Data from HEAD-NECK-RADIOMICS-HN1 [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/tcia.2019.8kap372n
- **License**:
    - Images: [TCIA No Commercial Limited](https://www.cancerimagingarchive.net/wp-content/uploads/TCIA-License-for-Limited-Access-Collections-w-NC-Final20220121.pdf)
    - Clinical: [CC BY-NC 3.0](https://creativecommons.org/licenses/by-nc/3.0/)
- **Data Types**: 
    - Images: CT, RTSTRUCT
    - Clinical: CSV
- **Sample Size**: 137 subjects


## RADCURE
- **Name**: RADCURE
- **Version/Date**: Clinical: Version 4: Updated 2024/12/19 / Images: Version 3: Updated 2024/03/27
- **URL**: <https://www.cancerimagingarchive.net/collection/radcure/>
- **Access Method**: NBIA Data Retriever
- **Access Date**: 2024-05-23
- **Data Format**: DICOM
- **Citation**: Welch, M. L., Kim, S., Hope, A., Huang, S. H., Lu, Z., Marsilla, J., Kazmierski, M., Rey-McIntyre, K., Patel, T., O’Sullivan, B., Waldron, J., Kwan, J., Su, J., Soltan Ghoraie, L., Chan, H. B., Yip, K., Giuliani, M., Princess Margaret Head And Neck Site Group, Bratman, S., … Tadic, T. (2023). Computed Tomography Images from Large Head and Neck Cohort (RADCURE) (Version 4) [Dataset]. The Cancer Imaging Archive. https://doi.org/10.7937/J47W-NM11
- **License**:
    - Images: [TCIA Restricted](https://wiki.cancerimagingarchive.net/download/attachments/4556915/TCIA%20Restricted%20License%2020220519.pdf?api=v2)
    - Clinical: [CC BY-NC 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Data Types**: 
    - Images: CT, RTSTRUCT
    - Clinical: CSV
- **Sample Size**: 3,346 subjects
---
## Overview

This section should document all data sources used in your project.
Proper documentation ensures reproducibility and helps others
understand your research methodology.

## How to Document Your Data

For each data source, include the following information:

### 1. External Data Sources

- **Name**: Official name of the dataset
- **Version/Date**: Version number or access date
- **URL**: Link to the data source
- **Access Method**: How the data was obtained (direct download, API, etc.)
- **Access Date**: When the data was accessed/retrieved
- **Data Format**: Format of the data (FASTQ, DICOM, CSV, etc.)
- **Citation**: Proper academic citation if applicable
- **License**: Usage restrictions and attribution requirements

Example:

```markdown
## TCGA RNA-Seq Data

- **Name**: The Cancer Genome Atlas RNA-Seq Data
- **Version**: Data release 28.0 - March 2021
- **URL**: https://portal.gdc.cancer.gov/
- **Access Method**: GDC Data Transfer Tool
- **Access Date**: 2021-03-15
- **Citation**: The Cancer Genome Atlas Network. (2012). Comprehensive molecular portraits of human breast tumours. Nature, 490(7418), 61-70.
- **License**: [NIH Genomic Data Sharing Policy](https://sharing.nih.gov/genomic-data-sharing-policy)
```

### 2. Internal/Generated Data

- **Name**: Descriptive name of the dataset
- **Creation Date**: When the data was generated
- **Creation Method**: Brief description of how the data was created
- **Input Data**: What source data was used
- **Processing Scripts**: References to scripts/Github Repo used to generate this data

Example:

```markdown
## Processed RNA-Seq Data
- **Name**: Processed RNA-Seq Data for TCGA-BRCA
- **Creation Date**: 2021-04-01
- **Creation Method**: Processed using kallisto and DESeq2
- **Input Data**: FASTQ Data obtained from the SRA database
- **Processing Scripts**: [GitHub Repo](https://github.com/tcga-brca-rnaseq)
```

### 3. Data Dictionary

For complex datasets, include a data dictionary that explains:

| Column Name | Data Type | Description | Units | Possible Values |
|-------------|-----------|-------------|-------|-----------------|
| patient_id  | string    | Unique patient identifier | N/A | TCGA-XX-XXXX format |
| age         | integer   | Patient age at diagnosis | years | 18-100 |
| expression  | float     | Gene expression value | TPM | Any positive value |

## Best Practices

- Store raw data in `data/rawdata/` and never modify it
- Store processed data in `data/procdata/` and all code used to generate it should be in `workflow/scripts/`
- Document all processing steps
- Track data provenance (where data came from and how it was modified)
- Respect data usage agreements and licenses!
    This is especially important for data that should not be shared publicly
