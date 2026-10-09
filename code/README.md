# PD-MCI nomogram: Main analysis code

This directory contains the manuscript's primary experimental workflow: cohort preparation, subject-level splitting, imputation, descriptive statistics, correlation screening, LASSO selection, the final logistic regression, eight benchmark models, clustered bootstrap confidence intervals, comparative ROC/PR plots, and the R nomogram.

The public package excludes sensitivity analyses, alternative analysis branches, presentation-only redraw scripts, manuscript build tools, and local review records.

## Data and configuration

Obtain `PPMI_Curated_Data_Cut_Public_20260810.xlsx` through the [PPMI data access portal](https://www.ppmi-info.org/access-data-specimens/download-data), subject to PPMI registration and data-use terms, and place it in this directory. Raw data, derived participant/visit-level datasets, and individual predictions are not distributed.

The three included CSV files are feature configuration only:

- `PPMI_initial_feature_sets.csv`: extraction and initial screening configuration.
- `PPMI_feature_sets.csv`: cohort feature selection configuration.
- `PPMI_feature_mapping.csv`: original feature names, abbreviations, and display names.

The workflow uses the curated cut directly, without cross-table backfilling. Cohort assertions are specific to this cut; a different cut can legitimately fail those assertions and is not an exact reproduction.

## Software and execution

The recorded analysis environment is Python 3.13.5 with the pinned packages in `requirements.txt`, and R 4.6.1 with `rms` 8.1.1 and `jsonlite` 2.0.0. `pdfcrop` is optional for trimming the nomogram PDF.

From the repository root, install the Python dependencies:

```bash
python -m pip install -r code/requirements.txt
```

Install the required R packages separately. Inspect the execution plan before running:

```bash
python code/run_all.py --dry-run
python code/run_all.py
```

The runner changes to its own directory, executes an explicit 23-script Python/R manifest, and stops on the first failed step. It also supports `--python-only`, `--r-only`, `--start-from`, and `--skip`; later steps require outputs from earlier steps. The frozen full run took approximately 160 minutes, mainly because of the eight benchmark hyperparameter searches. Running the workflow writes intermediate datasets and results locally.

`utils.py` and `figstyle.py` provide shared evaluation and plotting functions. `model_parameters.json` is the single frozen six-predictor model artifact used by the analysis, R nomogram, and browser calculator. The repository-root `index.html` loads it directly from `code/model_parameters.json`. Re-running model fitting can overwrite this artifact; synchronize the calculator's embedded defaults when intentionally publishing a changed model. Bootstrap inference is part of the primary model evaluation, rather than an optional sensitivity-analysis branch.

## Scope of the model

The outcome is the concurrent PPMI-derived operational cognitive-test PD-MCI indicator. Motor predictors use OFF-medication assessments in treated participants or assessments in untreated participants. The model supports exploratory visit-level risk stratification and is not a standalone clinical diagnosis or a prospective conversion model.

Python files use Black formatting with an 88-character line target. Code comments, docstrings, terminal messages, warnings, and explanatory output labels are in English. Formatting and language changes were checked to preserve computational structure and interpolated expressions. The existing frozen analyses were not rerun for this presentation-only cleanup.

Repository: [yuzhounh/PD-MCI-Nomogram](https://github.com/yuzhounh/PD-MCI-Nomogram).
