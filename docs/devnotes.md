# Developer Notes

## Data Processing Notes
[2025-05-05] PyRadiomics original_all_features extraction tracking
[2025-05-07] Updated with completed HN1 runs

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


[2025-05-12] Tracking extraction of all features for Aerts signature
|Feature / Dataset                        | HEAD-NECK-RADIOMICS-HN1 | NSCLC-Radiomics | RADCURE |
|original_firstorder_Energy               |             X           |        X        |    X    |
|original_shape_Compactness1              |             X           |        X        |    X    |
|original_glrlm_GrayLevelNonUniformity    |             X           |        X        |    X    |
|wavelet-HLH_glrlm_GrayLevelNonUniformity |          running        |   running       |    X    |

* Bootstrapping help came from: https://acclab.github.io/bootstrap-confidence-intervals.html
* survcomp R package only works for linux and osx-64
* tried the scikit-survival implementation of the concordance index with bootstrapping, but results don't match Mattea's exactly
* trying with R now

## Purpose of This Section

This section is for documenting technical decisions, challenges, and solutions encountered during your project. These notes are valuable for:

- Future you (who will forget why certain decisions were made)
- Collaborators who join the project later
- People coming from your publication who want to reproduce your work
- Anyone who might want to extend your research

## What to Document

### Design Decisions

Document important decisions about your project's architecture, algorithms, or methodologies:

``` markdown
## Choice of RNA-Seq Analysis Pipeline

[2025-04-25] We chose the kallisto over STAR pipeline for the following reasons:
    1. The CCLE dataset is very large, and kallisto is faster for quantifying large datasets
    2. GDSC used kallisto, so we can compare our results with theirs
```

### Technical Challenges

Record significant problems you encountered and how you solved them

``` markdown
## Sample Name Format Issue

[2025-04-25] We encountered a problem with sample name formats between the CCLE and GDSC datasets.
    The CCLE dataset uses "BRCA-XX-XXXX" format, while the GDSC dataset uses "BRCA-XX-XXXX-XX".
    We had to write a script to remove the last two characters from the sample names in the GDSC dataset.
```

### Dependencies and Environment

Document specific version requirements or compatibility issues:

``` markdown
## Critical Version Dependencies

[2025-04-25] SimpleITK 2.4.1 introduced a bug that flips images, so we froze version 2.4.0
```

## Best Practices

- Date your entries when appropriate
- Link to relevant code files or external resources
- Include small code snippets when helpful
- Note alternatives you considered and why they were rejected
- Document failed approaches to prevent others from repeating mistakes
- Update notes when major changes are made to the approach


