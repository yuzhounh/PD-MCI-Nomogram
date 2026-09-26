# PD-MCI Nomogram

A browser-based interface for exploring a logistic regression model of mild cognitive impairment in Parkinson's disease.

[MIT license](LICENSE)

## Quick Start

Open [index.html](index.html) in a browser with JavaScript enabled. No build command or application server is required by the current source.

The page loads Lucide icons from jsDelivr, so rendering those icons requires network access. The model calculations and charts are implemented in the page's own JavaScript.

## Usage

Enter sex, years of education, RBDSQ, GDS, SCOPA-AUT, Hoehn and Yahr stage, PIGD, and UPDRS-III scores. The interface updates the model probability, contribution chart, and threshold-based explanation.

The source defines three thresholds: standard, F1, and Youden. Model coefficients and thresholds are stored in the `MODEL` constant in [index.html](index.html).

The page states that the tool is intended for research and clinical reference only, not as the sole diagnostic basis. The repository does not include a training dataset, model-fitting code, or a study citation establishing the displayed model's validation.

## Repository Structure

- [index.html](index.html): interface, styles, logistic model, charts, and interaction handlers.
- [LICENSE](LICENSE): MIT license.

## License

See the existing [MIT license](LICENSE).

Copyright (c) 2026 Jing Wang.

