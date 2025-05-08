# Raw Data Directory

## Purpose

This directory is for storing **immutable raw data files** that serve as the original source data for your project. Raw data should never be modified after being placed here.

## IMPORTANT: Documentation Requirement

**EVERY TIME** you add data to this directory, you **MUST** document it in `docs/data_sources.md`. This documentation is critical for:

- Ensuring transparency in data acquisition
- Allowing others to reproduce your work
- Tracking data provenance and versioning
- Understanding special processing requirements or limitations

## Git Synchronization Notice

**⚠️ FILES IN THIS DIRECTORY ARE NOT SYNCHRONIZED WITH GIT ⚠️**

Raw data files are typically large and should not be stored in version control. Instead:

- Only the directory structure and this README are tracked
- You are responsible for backing up your data appropriately
- Consider using symbolic links to reference data from your project directories

## README Requirements for Data Subdirectories

When creating subdirectories, and possibly adding files here for your project,
**YOU MUST** add a section to the `docs/data_sources.md` file that will help
document the data you used and where it came from.

See the `docs/data_sources.md` file for more details and examples.

Remember that proper data management is essential for research reproducibility and collaboration!
