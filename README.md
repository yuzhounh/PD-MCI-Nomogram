# PD-MCI Nomogram

Primary experimental code and an interactive web calculator for a six-predictor nomogram estimating the probability of concurrent mild cognitive impairment in Parkinson's disease (PD-MCI), using the PPMI-derived operational cognitive-test indicator.

- **Use the calculator:** [Open the web calculator](https://yuzhounh.github.io/PD-MCI-Nomogram/).
- **Explore the analysis:** [Main experimental code](code/).
- **Reproduce the workflow:** [Detailed analysis instructions](code/README.md).

## Project overview

The analysis develops and internally validates an interpretable logistic regression model using the Parkinson's Progression Markers Initiative (PPMI) curated data cut of 10 August 2026. The workflow includes subject-level separation of training and validation sets, imputation, feature selection, model fitting, and evaluation against eight benchmark algorithms.

The web calculator provides a browser interface to the frozen final model. It requires six inputs: sex, years of education, Geriatric Depression Scale (GDS) score, modified Hoehn and Yahr (H&Y) stage, postural instability and gait difficulty (PIGD) score, and MDS-UPDRS Part III score. Motor assessments refer to OFF-medication assessments in treated participants or assessments in untreated participants.

The modeled endpoint is a concurrent, operational cognitive-test indicator. The calculator supports exploratory visit-level risk stratification; it does not establish a clinical diagnosis or predict future conversion to PD-MCI.

## Main experimental code

The [`code/`](code/) directory contains the primary experiments reported in the manuscript:

1. **Prepare the cohort:** Extract the curated input, assess initial missingness, apply cohort and feature rules, split subjects into training and validation sets, derive disease duration, and impute predictors.
2. **Describe and select features:** Summarize visit-level characteristics, screen correlations, remove redundant features, and select predictors with LASSO.
3. **Fit and compare models:** Fit the final unpenalized logistic regression and benchmark CatBoost, EasyEnsemble, LightGBM, tuned logistic regression, multilayer perceptron, random forest, support vector machine, and XGBoost models.
4. **Evaluate and visualize:** Calculate patient-cluster-bootstrap confidence intervals, plot comparative ROC/PR curves, and generate the R nomogram from the full-precision final model coefficients.

[`code/run_all.py`](code/run_all.py) runs an explicit manifest of 23 Python/R scripts and stops at the first failed step. [`code/utils.py`](code/utils.py) and [`code/figstyle.py`](code/figstyle.py) provide shared evaluation and plotting functions. The feature configuration CSV files define extraction, selection, and display names; they contain no participant-level observations.

The public package contains the main experimental workflow and its required configuration. Sensitivity analyses, alternative analysis branches, presentation-only redraw scripts, manuscript build tools, and local review records are outside this package. Clustered bootstrap inference is part of the primary model evaluation.

### Data access

To rerun the analysis, obtain `PPMI_Curated_Data_Cut_Public_20260810.xlsx` through the [PPMI data access portal](https://www.ppmi-info.org/access-data-specimens/download-data), following PPMI registration and data-use requirements, and place it in `code/`.

Raw PPMI data, derived participant/visit-level datasets, and individual predictions are not distributed in this repository. The workflow uses this curated cut directly, without cross-table backfilling. Cohort assertions are specific to the recorded cut; a different data cut is not an exact reproduction.

### Run the analysis

The recorded environment is Python 3.13.5 with pinned dependencies in [`code/requirements.txt`](code/requirements.txt), and R 4.6.1 with `rms` 8.1.1 and `jsonlite` 2.0.0. Install the Python dependencies and the required R packages before running the workflow.

From the repository root:

```bash
python -m pip install -r code/requirements.txt
python code/run_all.py --dry-run
python code/run_all.py
```

The runner changes to `code/` automatically. It supports `--python-only`, `--r-only`, `--start-from`, and `--skip`; later steps require the outputs of earlier steps. The recorded complete run took approximately 160 minutes, mainly because of benchmark hyperparameter searches. Outputs are written locally under `code/`.

See [`code/README.md`](code/README.md) for the feature configuration, execution details, and model-file behavior.

## Web calculator

The [online calculator](https://yuzhounh.github.io/PD-MCI-Nomogram/) runs in the browser and requires no access to the PPMI dataset or installation of the analysis packages.

Enter all six predictors to view the estimated probability, its position relative to the selected threshold, and each predictor's signed contribution to the model linear predictor. The interface does not impute missing inputs. The three operating thresholds are:

- **Default:** 0.5000.
- **Youden:** 0.1966, derived from the training data.
- **F1:** 0.1977, derived from the training data.

The training-derived thresholds are exploratory references, not externally validated clinical cutoffs. A negative operational indicator does not establish clinically normal cognition.

[`index.html`](index.html) contains the interface, styles, charts, and browser calculation logic. It loads the full-precision intercept, coefficients, and thresholds directly from [`code/model_parameters.json`](code/model_parameters.json), with matching embedded defaults if the JSON cannot be loaded. Lucide icons are loaded from jsDelivr and require network access.

### Run the calculator locally

From the root of a checkout of this public repository:

```bash
python -m http.server 8000
```

Then open [http://localhost:8000/](http://localhost:8000/) in a browser. Serving over HTTP allows the page to load its parameter JSON.

## Repository structure and GitHub Pages

```text
PD-MCI-Nomogram/
├── README.md                    # Project overview: analysis and calculator
├── LICENSE                      # MIT license
├── index.html                   # GitHub Pages calculator entry point
└── code/
    ├── README.md                # Detailed analysis instructions
    ├── requirements.txt        # Pinned Python analysis dependencies
    ├── run_all.py              # Main workflow runner
    ├── step_*.py               # Primary Python experiments and evaluation
    ├── step_16_nomogram.R      # Nomogram generation
    ├── utils.py                # Shared evaluation functions
    ├── figstyle.py             # Shared plotting style
    ├── PPMI_*feature*.csv      # Feature configuration, not participant data
    └── model_parameters.json   # Shared model artifact for analysis and calculator
```

GitHub Pages currently publishes `main` from the repository root (`/`). Keeping `index.html` at that publishing root makes the calculator available at the existing project URL. The calculator needs no application build command or backend service.

[`code/model_parameters.json`](code/model_parameters.json) is the single model-parameter file shared by the analysis, R nomogram, and browser calculator. The page loads it using the relative path `code/model_parameters.json`; no duplicate parameter file is kept at the repository root. Running the model-fitting pipeline can overwrite this artifact, so synchronize the calculator's embedded defaults when intentionally publishing a changed model.

The root README introduces the complete project; `code/README.md` provides the directory-specific reproducibility instructions. Having both supports separate browsing of the project and its analysis code.

## License

The code is distributed under the existing [MIT license](LICENSE). PPMI data access and reuse remain subject to the applicable PPMI terms.
