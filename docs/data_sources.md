# Data Sources

This section documents all data sources used in READII-2-ROQC.

## External Data Sources

### CC-Radiomics-Phantom
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


### NSCLC-Radiomics
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


### HEAD-NECK-RADIOMICS-HN1
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
- **ROI Name**: Tumour = GTV-1


### RADCURE
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
- **ROI Name**: Tumour = GTVp$ (regex to just get first primary tumour)

---

## Internal/Generated Data


-- 

<!--
### Data Dictionary

For complex datasets, include a data dictionary that explains:

| Column Name | Data Type | Description | Units | Possible Values |
|-------------|-----------|-------------|-------|-----------------|
| patient_id  | string    | Unique patient identifier | N/A | TCGA-XX-XXXX format |
| age         | integer   | Patient age at diagnosis | years | 18-100 |
| expression  | float     | Gene expression value | TPM | Any positive value |
-->

