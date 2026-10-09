import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)
import os

import figstyle as fs

fs.apply()
plt.rcParams["axes.unicode_minus"] = True
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Arial"
plt.rcParams["mathtext.it"] = "Arial:italic"
plt.rcParams["mathtext.bf"] = "Arial:bold"

# Define model colors and display names using common abbreviations.
model_info = {
    "lr": {"name": "LR-tuned", "color": "#E74C3C"},
    "svm": {"name": "SVM", "color": "#3498DB"},
    "rf": {"name": "RF", "color": "#2ECC71"},
    "xgboost": {"name": "XGBoost", "color": "#9B59B6"},
    "lightgbm": {"name": "LightGBM", "color": "#1ABC9C"},
    "catboost": {"name": "CatBoost", "color": "#F39C12"},
    "mlp": {"name": "MLP", "color": "#E91E63"},
    "easyensemble": {"name": "EasyEnsemble", "color": "#00BCD4"},
}


def load_predictions(model_name):
    """Load model predictions."""
    file_path = f"step_14_{model_name}_predictions.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        return df["y_true"].values, df["y_prob"].values
    return None, None


def plot_multi_roc_pr_curves(models, save_path="step_15_roc_pr_curves.pdf"):
    """Plot comparative ROC and PR curves for multiple models and save as PDF."""
    import os

    fig, axes_grid = fs.axes_grid(1, 2)
    ax1, ax2 = axes_grid[0]
    h1, n1, v1, h2, n2, v2 = [], [], [], [], [], []

    last_y_true = None  # Used for the baseline.

    # ROC curves.
    for model_name in models:
        y_true, y_prob = load_predictions(model_name)
        if y_true is None:
            continue

        last_y_true = y_true  # Retain the last non-None y_true.

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_roc = roc_auc_score(y_true, y_prob)
        info = model_info.get(model_name, {"name": model_name, "color": "gray"})

        h1.append(ax1.plot(fpr, tpr, color=info["color"], linewidth=1.2)[0])
        n1.append(info["name"])
        v1.append(f"{auc_roc:.4f}")

    # ROC Baseline
    h1.append(ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=0.9)[0])
    n1.append("Random")
    v1.append(f"{0.5:.4f}")
    ax1.set_xlabel("1 - Specificity")
    ax1.set_ylabel("Sensitivity")
    fs.table_legend(
        ax1, h1, n1, [("AUROC", v1)], "Model", fontsize=5.6, loc="lower right"
    )
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])
    fs.panel_letter(ax1, "A")

    # PR curves.
    for model_name in models:
        y_true, y_prob = load_predictions(model_name)
        if y_true is None:
            continue

        last_y_true = y_true

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        auc_pr = average_precision_score(y_true, y_prob)
        info = model_info.get(model_name, {"name": model_name, "color": "gray"})

        h2.append(ax2.plot(recall, precision, color=info["color"], linewidth=1.2)[0])
        n2.append(info["name"])
        v2.append(f"{auc_pr:.4f}")

    # Baseline
    if last_y_true is not None:
        baseline = np.mean(last_y_true)
        h2.append(
            ax2.axhline(y=baseline, color="k", linestyle="--", alpha=0.5, linewidth=0.9)
        )
        n2.append("Random")
        v2.append(f"{baseline:.4f}")

    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    fs.table_legend(
        ax2, h2, n2, [("AUPRC", v2)], "Model", fontsize=5.6, loc="upper right"
    )
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([-0.02, 1.02])
    fs.panel_letter(ax2, "B")

    # Save as PDF.
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Comparison ROC and PR curves saved to: {save_path}")


if __name__ == "__main__":
    # Select models for comparison.
    models_to_compare = [
        "lr",
        "svm",
        "rf",
        "xgboost",
        "lightgbm",
        "catboost",
        "mlp",
        "easyensemble",
    ]

    # Exclude models whose prediction files do not exist.
    valid_models = []
    for model in models_to_compare:
        if os.path.exists(f"step_14_{model}_predictions.csv"):
            valid_models.append(model)
        else:
            print(f"Warning: step_14_{model}_predictions.csv not found, skipping...")

    if valid_models:
        plot_multi_roc_pr_curves(valid_models)
    else:
        print("No valid prediction files found!")
