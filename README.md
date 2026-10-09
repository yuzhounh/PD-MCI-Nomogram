# PD-MCI Nomogram

A browser-based calculator for exploratory, visit-level estimation of concurrent mild cognitive impairment probability in Parkinson's disease, using the PPMI operational cognitive-test indicator.

Use the calculator at [GitHub Pages](https://yuzhounh.github.io/PD-MCI-Nomogram/).

## Inputs and calculation

The six inputs are sex, completed years of education, Geriatric Depression Scale score, modified Hoehn and Yahr stage, postural instability and gait difficulty score, and MDS-UPDRS Part III score. Motor assessments must be OFF-medication in treated participants or obtained in untreated participants. All six inputs are required; the interface does not impute missing values.

The probability is calculated from the full-precision intercept and coefficients in [model_parameters.json](model_parameters.json), with matching embedded defaults in [index.html](index.html). The thresholds are 0.5000 (default), 0.1966 (training-derived Youden), and 0.1977 (training-derived F1). The latter two are exploratory references, not externally validated clinical cutoffs.

The page shows the estimated probability, its relation to the selected threshold, and each predictor's signed contribution to the model linear predictor. A negative operational indicator does not establish clinically normal cognition. This tool is for research and exploratory risk stratification and must not be used as a standalone diagnostic or exclusion tool.

## Files and hosting

- `index.html`: interface, styles, charts, and calculation logic.
- `model_parameters.json`: model coefficients, thresholds, and aggregate validation information.
- `README.md`: calculator description.

GitHub Pages serves the repository root on `main`; no build command is needed. For local use, serve these files over HTTP so the parameter JSON can load. Lucide icons load from jsDelivr and require network access. The repository's existing MIT license remains applicable.
