# Data Sources

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
