# Results Directory

## Purpose

This directory stores the **output files** from your analyses, including:

- Final data matrices
- Statistical analysis results
- Summary tables
- If using other software for organizing figures/tables, consider including references to them
in the repository documentation so they dont get lost.
  - tip: [draw.io](https://app.diagrams.net/) lets you version control diagrams
  in git, and you can export them as images for publication.

## Documentation Best Practices

While detailed documentation is always valuable, results should be traceable through your analysis code. Make sure that:

- Your analysis code (in `workflow/scripts` or `workflow/notebooks`) clearly documents how results were generated
- Results files have descriptive names that indicate what they contain
  - i.e use `gene_expression_analysis_results.csv` instead of `results.csv`
  - i.e use `2025-05-08_features-summary.png` instead of `summary.png`
- **Focus on versioning the code that generates the results rather than the results themselves**

## Git Synchronization Notice

**⚠️ FILES IN THIS DIRECTORY ARE NOT SYNCHRONIZED WITH GIT ⚠️**

Result files can vary in size but are often too large for version control. Instead:

- Only the directory structure and this README are tracked
- Smaller result files might be suitable for Git, but consider using other means for larger files
- **Focus on versioning the code that generates the results rather than the results themselves**

## Organization Recommendations

Consider organizing results by:

- Analysis date
- Analysis type
- Project milestone
- Figure/table number (if results directly correspond to publication elements)

Example structure:

```console
/results/pathway_analysis/
/results/figures/
/results/tables/
```

Remember that properly organized results make it easier to find what you need when writing papers or preparing presentations!
