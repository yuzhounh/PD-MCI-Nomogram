"""Patient-clustered bootstrap confidence intervals for benchmark models.

This script is a post-processing step. It uses the saved subject-held-out
validation predictions and fixed training-derived thresholds; it does not fit,
tune, or otherwise rerun any benchmark model.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    roc_auc_score,
)


BASE_DIR = Path(__file__).resolve().parent
N_BOOTSTRAP = 2000
RANDOM_SEED = 42

MODELS = [
    ("lr", "LR-tuned"),
    ("svm", "SVM"),
    ("rf", "RF"),
    ("xgboost", "XGBoost"),
    ("lightgbm", "LightGBM"),
    ("catboost", "CatBoost"),
    ("mlp", "MLP"),
    ("easyensemble", "EasyEnsemble"),
]

THRESHOLD_COLUMNS = [
    ("Default", "test_default"),
    ("F1-optimal", "test_f1"),
    ("Youden-optimal", "test_youden"),
]

INDEPENDENT_METRICS = ["AUROC", "AUPRC", "Brier score"]
DEPENDENT_METRICS = [
    "Accuracy",
    "Balanced accuracy",
    "Precision",
    "NPV",
    "Sensitivity",
    "Specificity",
    "F1-score",
    "Cohen's Kappa",
]


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else np.nan


def threshold_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = safe_ratio(tp, tp + fn)
    specificity = safe_ratio(tn, tn + fp)
    precision = safe_ratio(tp, tp + fp)
    npv = safe_ratio(tn, tn + fn)
    return {
        "Accuracy": safe_ratio(tp + tn, tp + tn + fp + fn),
        "Balanced accuracy": np.nanmean([sensitivity, specificity]),
        "Precision": precision,
        "NPV": npv,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "F1-score": safe_ratio(2 * tp, 2 * tp + fp + fn),
        "Cohen's Kappa": cohen_kappa_score(y_true, y_pred),
    }


def independent_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "AUROC": roc_auc_score(y_true, y_prob),
        "AUPRC": average_precision_score(y_true, y_prob),
        "Brier score": brier_score_loss(y_true, y_prob),
    }


def percentile_summary(values: list[float]) -> tuple[float, float, int]:
    valid = np.asarray(values, dtype=float)
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        return np.nan, np.nan, 0
    return (
        float(np.percentile(valid, 2.5)),
        float(np.percentile(valid, 97.5)),
        int(valid.size),
    )


def build_cluster_samples(subjects: np.ndarray, y_true: np.ndarray) -> list[np.ndarray]:
    unique_subjects = np.unique(subjects)
    subject_indices = {s: np.flatnonzero(subjects == s) for s in unique_subjects}
    rng = np.random.RandomState(RANDOM_SEED)
    samples = []
    while len(samples) < N_BOOTSTRAP:
        sampled_subjects = rng.choice(
            unique_subjects, size=len(unique_subjects), replace=True
        )
        indices = np.concatenate([subject_indices[s] for s in sampled_subjects])
        if np.unique(y_true[indices]).size == 2:
            samples.append(indices)
    return samples


def format_ci(row: pd.Series) -> str:
    return (
        f"{row['Point_Estimate']:.4f} "
        f"({row['CI_Lower']:.4f}--{row['CI_Upper']:.4f})"
    )


def latex_rows(results: pd.DataFrame, models: list[str]) -> str:
    by_key = results.set_index(["Model", "Threshold_Setting", "Metric"])
    n_columns = len(models) + 1
    lines = []
    for metric in INDEPENDENT_METRICS:
        cells = [
            format_ci(by_key.loc[(model, "Independent", metric)]) for model in models
        ]
        lines.append(rf"{metric} & " + " & ".join(cells) + r" \\")

    for setting, _ in THRESHOLD_COLUMNS:
        lines.append(r"\addlinespace[0.35em]")
        label = (
            "Default threshold (0.50)"
            if setting == "Default"
            else f"{setting} threshold"
        )
        lines.append(
            rf"\rowcolor{{gray!15}} \multicolumn{{{n_columns}}}{{l}}{{\textbf{{{label}}}}} \\"
        )
        if setting != "Default":
            thresholds = []
            for model in models:
                subset = results[
                    (results["Model"] == model)
                    & (results["Threshold_Setting"] == setting)
                ]
                thresholds.append(f"{subset['Threshold'].iloc[0]:.4f}")
            lines.append(r"Threshold & " + " & ".join(thresholds) + r" \\")
        for metric in DEPENDENT_METRICS:
            cells = [
                format_ci(by_key.loc[(model, setting, metric)]) for model in models
            ]
            latex_metric = {
                "Balanced accuracy": "BAcc",
                "Cohen's Kappa": "Kappa",
            }.get(metric, metric)
            lines.append(rf"{latex_metric} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines) + "\n"


def main() -> None:
    validation = pd.read_csv(BASE_DIR / "PPMI_9_test.csv")
    y_reference = validation["MCI"].to_numpy(dtype=int)
    subjects = validation["PATNO"].to_numpy()
    samples = build_cluster_samples(subjects, y_reference)

    rows = []
    for model_key, model_name in MODELS:
        predictions = pd.read_csv(BASE_DIR / f"step_14_{model_key}_predictions.csv")
        if len(predictions) != len(validation):
            raise ValueError(
                f"{model_name}: prediction row count does not match validation data"
            )
        y_true = predictions["y_true"].to_numpy(dtype=int)
        y_prob = predictions["y_prob"].to_numpy(dtype=float)
        if not np.array_equal(y_true, y_reference):
            raise ValueError(f"{model_name}: saved labels do not match PPMI_9_test.csv")

        performance = pd.read_csv(
            BASE_DIR / f"step_14_{model_key}_test_performance.csv", index_col=0
        )
        thresholds = {
            setting: float(performance.loc["Optimal Threshold", column])
            for setting, column in THRESHOLD_COLUMNS
        }

        point_independent = independent_metrics(y_true, y_prob)
        bootstrap_independent = {metric: [] for metric in INDEPENDENT_METRICS}
        bootstrap_dependent = {
            (setting, metric): []
            for setting, _ in THRESHOLD_COLUMNS
            for metric in DEPENDENT_METRICS
        }

        for indices in samples:
            y_true_b = y_true[indices]
            y_prob_b = y_prob[indices]
            for metric, value in independent_metrics(y_true_b, y_prob_b).items():
                bootstrap_independent[metric].append(value)
            for setting, _ in THRESHOLD_COLUMNS:
                values = threshold_metrics(y_true_b, y_prob_b, thresholds[setting])
                for metric, value in values.items():
                    bootstrap_dependent[(setting, metric)].append(value)

        for metric in INDEPENDENT_METRICS:
            lower, upper, n_valid = percentile_summary(bootstrap_independent[metric])
            rows.append(
                {
                    "Model": model_name,
                    "Threshold_Setting": "Independent",
                    "Threshold": np.nan,
                    "Metric": metric,
                    "Point_Estimate": point_independent[metric],
                    "CI_Lower": lower,
                    "CI_Upper": upper,
                    "N_Bootstrap": n_valid,
                }
            )

        for setting, _ in THRESHOLD_COLUMNS:
            point_values = threshold_metrics(y_true, y_prob, thresholds[setting])
            for metric in DEPENDENT_METRICS:
                lower, upper, n_valid = percentile_summary(
                    bootstrap_dependent[(setting, metric)]
                )
                rows.append(
                    {
                        "Model": model_name,
                        "Threshold_Setting": setting,
                        "Threshold": thresholds[setting],
                        "Metric": metric,
                        "Point_Estimate": point_values[metric],
                        "CI_Lower": lower,
                        "CI_Upper": upper,
                        "N_Bootstrap": n_valid,
                    }
                )

        print(f"Completed {model_name}")

    results = pd.DataFrame(rows)
    numeric_columns = ["Threshold", "Point_Estimate", "CI_Lower", "CI_Upper"]
    results[numeric_columns] = results[numeric_columns].round(4)
    results.to_csv(BASE_DIR / "step_14_bootstrap_ci.csv", index=False)

    all_models = [name for _, name in MODELS]
    (BASE_DIR / "step_14_bootstrap_ci_table.tex").write_text(
        latex_rows(results, all_models), encoding="utf-8"
    )

    print(f"Saved {len(results)} metric rows with {N_BOOTSTRAP} cluster resamples")


if __name__ == "__main__":
    main()
