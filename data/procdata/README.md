# Processed Data Directory

## Purpose

This directory stores **pre-processed data files** that have been derived from raw data but are still intended for use across multiple analyses. Examples include:

- Normalized data matrices
- Filtered datasets
- Aligned sequences
- Feature extracted data

## IMPORTANT: Documentation Requirement

When adding processed data to this directory, you **MUST** document:

1. The source of the raw data (referencing `docs/data_sources.md`)
2. The processing steps or pipeline used
3. Version information for tools used in processing

This documentation ensures research reproducibility and transparency in your data processing workflow.

## Git Synchronization Notice

**⚠️ FILES IN THIS DIRECTORY ARE NOT SYNCHRONIZED WITH GIT ⚠️**

Processed data files are typically too large for version control. Instead:

- Only the directory structure and this README are tracked
- Document your processing code in version control
(preferably in `workflow/scripts` or `workflow/notebooks`)!

Remember that well-documented processed data makes your research more
reproducible and accessible to collaborators!
