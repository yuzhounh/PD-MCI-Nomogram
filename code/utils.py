import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    brier_score_loss,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    cohen_kappa_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.inspection import permutation_importance
import shap
from matplotlib.font_manager import FontProperties

# Use the shared plotting style; see figstyle.py for fonts and panel dimensions.
import figstyle as fs

fs.apply()
plt.rcParams["axes.unicode_minus"] = True
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Arial"
plt.rcParams["mathtext.it"] = "Arial:italic"
plt.rcParams["mathtext.bf"] = "Arial:bold"


def load_data(train_file, test_file, feature_mapping_file):
    """Load training set, test set and feature mapping file"""
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    feature_mapping = pd.read_csv(feature_mapping_file)

    # Create feature name mapping dictionary
    feature_name_mapping = dict(
        zip(feature_mapping["feature"], feature_mapping["abbreviation"])
    )

    return train_data, test_data, feature_name_mapping


def prepare_data(data, subject_col="PATNO", label_col="MCI"):
    """Prepare data, separate features and labels"""
    # Remove subject ID column
    X = data.drop([subject_col, label_col], axis=1)
    y = data[label_col]
    subjects = data[subject_col]

    return X, y, subjects


def subject_level_cross_validation(X, y, subjects, n_folds=10, random_state=42):
    """
    Subject-level cross-validation - adopting "dimension reduction, splitting, dimension expansion" logic.
    """
    # Reduce to subject level
    subject_labels_series = pd.Series(y, index=subjects).groupby(level=0).max()
    unique_subjects = subject_labels_series.index.values
    subject_labels = subject_labels_series.values

    # Initialize stratified cross-validation
    sgkf = StratifiedGroupKFold(
        n_splits=n_folds, shuffle=True, random_state=random_state
    )

    folds = []

    # Complete splitting at subject level
    for train_subject_idx, val_subject_idx in sgkf.split(
        unique_subjects, subject_labels, groups=unique_subjects
    ):
        train_subjects_fold = unique_subjects[train_subject_idx]
        val_subjects_fold = unique_subjects[val_subject_idx]

        # Expand back to sample level
        train_sample_idx = np.where(np.isin(subjects, train_subjects_fold))[0]
        val_sample_idx = np.where(np.isin(subjects, val_subjects_fold))[0]

        folds.append((train_sample_idx, val_sample_idx))

    return folds


def f1_score_threshold(y_true, y_prob):
    """
    Calculate optimal threshold for maximum F1-score.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    f1_scores = np.nan_to_num(f1_scores)
    f1_scores = f1_scores[:-1]
    optimal_idx = np.argmax(f1_scores)
    return thresholds[optimal_idx]


def youden_threshold(y_true, y_prob):
    """Calculate optimal threshold based on Youden index"""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden_scores = tpr - fpr  # Youden's J statistic
    optimal_idx = np.argmax(youden_scores)
    return thresholds[optimal_idx]


def calculate_all_metrics(y_true, y_pred, y_prob, threshold=None):
    """Calculate all evaluation metrics"""
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cohen_kappa = cohen_kappa_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_prob)
    auc_pr = average_precision_score(y_true, y_prob)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Return results in specified order
    result = {
        "AUC-ROC": auc_roc,
        "AUC-PR": auc_pr,
        "Brier Score": brier_score_loss(y_true, y_prob),
        "Optimal Threshold": threshold if threshold is not None else 0.5,
        "Accuracy": accuracy,
        "Balanced Accuracy": balanced_accuracy,
        "Precision": precision,
        "NPV": npv,
        "Recall": recall,
        "Specificity": specificity,
        "F1-score": f1,
        "Cohen's Kappa": cohen_kappa,
    }

    return result


def save_figure(fig, save_path):
    """Save figure (PDF only)"""
    base, ext = os.path.splitext(save_path)
    if ext.lower() != ".pdf":
        save_path = base + ".pdf"
    fig.savefig(save_path, bbox_inches="tight")


def plot_roc_pr_curves(
    y_true,
    y_prob,
    save_path,
    model_name="Model",
    groups=None,
    n_bootstrap=2000,
    random_state=42,
):
    """Plot ROC and PR curves, optionally with clustered-bootstrap 95% CIs."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    roc_grid = np.linspace(0, 1, 201)
    pr_grid = np.linspace(0, 1, 201)
    roc_ci = pr_ci = None

    if groups is not None:
        groups = np.asarray(groups)
        if len(groups) != len(y_true):
            raise ValueError("groups must have the same length as y_true")

        unique_groups = np.unique(groups)
        group_indices = {
            group: np.flatnonzero(groups == group) for group in unique_groups
        }
        rng = np.random.RandomState(random_state)
        roc_samples = []
        pr_samples = []

        for _ in range(n_bootstrap):
            sampled_groups = rng.choice(
                unique_groups, size=len(unique_groups), replace=True
            )
            indices = np.concatenate([group_indices[group] for group in sampled_groups])
            y_boot = y_true[indices]
            p_boot = y_prob[indices]
            if len(np.unique(y_boot)) < 2:
                continue

            fpr_boot, tpr_boot, _ = roc_curve(y_boot, p_boot)
            roc_samples.append(np.interp(roc_grid, fpr_boot, tpr_boot))

            precision_boot, recall_boot, _ = precision_recall_curve(y_boot, p_boot)
            pr_samples.append(
                np.interp(pr_grid, recall_boot[::-1], precision_boot[::-1])
            )

        if roc_samples and pr_samples:
            roc_ci = np.percentile(np.asarray(roc_samples), [2.5, 97.5], axis=0)
            pr_ci = np.percentile(np.asarray(pr_samples), [2.5, 97.5], axis=0)

    fig, axes_grid = fs.axes_grid(1, 2)
    ax1, ax2 = axes_grid[0]

    # ── ROC curve ──────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_roc = roc_auc_score(y_true, y_prob)

    curve_color = "#1f77b4"
    h1 = [
        ax1.plot(fpr, tpr, color=curve_color, linewidth=1.4)[0],
        ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=0.9)[0],
    ]
    n1, v1 = [model_name, "Random"], [f"{auc_roc:.4f}", f"{0.5:.4f}"]
    if roc_ci is not None:
        ax1.fill_between(
            roc_grid,
            roc_ci[0],
            roc_ci[1],
            color=curve_color,
            alpha=0.18,
            linewidth=0,
        )
        h1.append(ax1.collections[-1])
        n1.append("95% CI")
        v1.append("")

    ax1.set_xlabel("1 - Specificity")
    ax1.set_ylabel("Sensitivity")
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])
    ax1.grid(True, alpha=0.3)
    fs.table_legend(ax1, h1, n1, [("AUROC", v1)], "Model", loc="lower right")
    fs.panel_letter(ax1, "A")

    # ── PR curve ───────────────────────────────────────────────────────────
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auc_pr = average_precision_score(y_true, y_prob)
    baseline = np.mean(y_true)

    h2 = [
        ax2.plot(recall, precision, color=curve_color, linewidth=1.4)[0],
        ax2.axhline(y=baseline, color="k", linestyle="--", alpha=0.5, linewidth=0.9),
    ]
    n2, v2 = [model_name, "Random"], [f"{auc_pr:.4f}", f"{baseline:.4f}"]
    if pr_ci is not None:
        ax2.fill_between(
            pr_grid,
            pr_ci[0],
            pr_ci[1],
            color=curve_color,
            alpha=0.18,
            linewidth=0,
        )
        h2.append(ax2.collections[-1])
        n2.append("95% CI")
        v2.append("")

    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([-0.02, 1.02])
    ax2.grid(True, alpha=0.3)
    fs.table_legend(ax2, h2, n2, [("AUPRC", v2)], "Model", loc="upper right")
    fs.panel_letter(ax2, "B")

    save_figure(fig, save_path)
    plt.close()


def plot_shap_summary(shap_values, X, feature_names, model_name, save_path):
    """Plot SHAP summary plots"""
    import os

    base_dir = os.path.dirname(save_path)
    base_name = os.path.splitext(os.path.basename(save_path))[0]

    # # Bar plot
    # fig = plt.figure(figsize=(10, 8))
    # shap.summary_plot(
    #     shap_values, X, feature_names=feature_names, plot_type="bar", show=False
    # )
    # plt.title(f"{model_name} - SHAP Feature Importance (Bar)")
    # plt.tight_layout()
    # save_figure(fig, os.path.join(base_dir, f"{base_name}_bar.pdf"))
    # plt.close()

    # Summary plot
    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.title(f"{model_name}")
    plt.tight_layout()
    save_figure(fig, os.path.join(base_dir, f"{base_name}_summary.pdf"))
    plt.close()

    # # Save SHAP values data
    # shap_df = pd.DataFrame(shap_values, columns=feature_names)
    # shap_df.to_csv(os.path.join(base_dir, f"{base_name}_values.csv"), index=False)


def get_permutation_importance(model, X, y, feature_names, random_state=42):
    """Calculate permutation importance"""
    perm_importance = permutation_importance(
        model, X, y, n_repeats=10, random_state=random_state, scoring="roc_auc"
    )
    return perm_importance.importances_mean


def print_results(model_name, metrics, optimal_threshold):
    """Print results"""
    print(f"\n{'='*50}")
    print(f"{model_name} Model Results")
    print(f"{'='*50}")
    print(f"Optimal Threshold: {optimal_threshold:.4f}")

    print(f"\nConfusion Matrix:")
    print(metrics["Confusion Matrix"])

    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        if metric != "Confusion Matrix":
            print(f"{metric}: {value}")


def plot_nomogram(coefficients, intercept, feature_names, save_path="nomogram.pdf"):
    """
    Plot a nomogram.

    Parameters:
        coefficients: Model coefficients.
        intercept: Intercept.
        feature_names: List of feature names.
        save_path: Output path.
    """
    fig, ax = plt.subplots(figsize=(12, len(feature_names) * 0.8 + 4))

    # Calculate the point range for each feature.
    n_features = len(feature_names)

    # Set the point range, assuming standardized features spanning approximately -3 to 3.
    feature_range = np.linspace(-3, 3, 100)

    # Create one row per feature.
    y_pos = np.arange(n_features + 2, 0, -1)

    # 1. Draw the total-points scale.
    ax.plot([0, 100], [y_pos[0], y_pos[0]], "k-", linewidth=2)
    for i in range(0, 101, 10):
        ax.plot([i, i], [y_pos[0] - 0.1, y_pos[0] + 0.1], "k-", linewidth=1)
        ax.text(i, y_pos[0] + 0.3, str(i), ha="center", fontsize=10)
    ax.text(50, y_pos[0] + 0.8, "Total Points", ha="center", fontsize=12)

    # 2. Draw the scale for each feature.
    max_abs_coef = np.max(np.abs(coefficients))

    for i, (coef, name) in enumerate(zip(coefficients, feature_names)):
        y = y_pos[i + 1]

        # Calculate this feature's point contribution, normalized to 0-100.
        points = (feature_range * coef / max_abs_coef) * 50 + 50

        # Draw the feature scale line.
        ax.plot([0, 100], [y, y], "k-", linewidth=1.5)

        # Draw tick marks.
        for j in range(0, 101, 20):
            # Find the corresponding feature value.
            feature_val = (j - 50) / 50 * max_abs_coef / coef if coef != 0 else 0
            ax.plot([j, j], [y - 0.1, y + 0.1], "k-", linewidth=1)
            ax.text(j, y - 0.4, f"{feature_val:.2f}", ha="center", fontsize=8)

        # Feature name.
        ax.text(-5, y, name, ha="right", va="center", fontsize=10)

    # 3. Draw the predicted-probability scale.
    y = y_pos[-1]
    ax.plot([0, 100], [y, y], "k-", linewidth=2)

    # Calculate points corresponding to each probability.
    prob_range = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    logit_range = np.log(prob_range / (1 - prob_range))

    # Normalize to 0-100.
    logit_min = np.log(0.01 / 0.99)
    logit_max = np.log(0.99 / 0.01)
    points_range = ((logit_range - logit_min) / (logit_max - logit_min)) * 100

    for prob, point in zip(prob_range, points_range):
        if 0 <= point <= 100:
            ax.plot([point, point], [y - 0.1, y + 0.1], "k-", linewidth=1)
            ax.text(point, y - 0.4, f"{prob:.1f}", ha="center", fontsize=8)

    ax.text(50, y + 0.8, "Predicted Probability", ha="center", fontsize=12)

    # Set figure properties.
    ax.set_xlim(-10, 105)
    ax.set_ylim(0, n_features + 3)
    ax.axis("off")
    ax.set_title("Nomogram for Logistic Regression", fontsize=14, pad=20)

    plt.tight_layout()
    save_figure(fig, save_path)
    plt.close()


def plot_decision_curve_analysis(y_true, y_prob, save_path="decision_curve.pdf"):
    """
    Plot decision curve analysis (DCA).

    Parameters:
        y_true: Observed outcome labels.
        y_prob: Predicted probabilities.
        save_path: Output path.
    """
    # Threshold-probability range.
    thresholds = np.linspace(0.01, 0.99, 100)

    net_benefits = []
    net_benefit_all = []

    for threshold in thresholds:
        # Convert probabilities to predicted classes.
        y_pred = (y_prob >= threshold).astype(int)

        # Count true positives and false positives.
        TP = np.sum((y_pred == 1) & (y_true == 1))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        n = len(y_true)

        # Calculate net benefit.
        net_benefit = (TP / n) - (FP / n) * (threshold / (1 - threshold))
        net_benefits.append(net_benefit)

        # Net benefit of treating all.
        prevalence = np.sum(y_true) / n
        net_benefit_treat_all = prevalence - (1 - prevalence) * (
            threshold / (1 - threshold)
        )
        net_benefit_all.append(net_benefit_treat_all)

    # Plot the figure.
    fig, ax = plt.subplots(figsize=(5, 4))

    # Plot the model decision curve.
    ax.plot(thresholds, net_benefits, label="Model", linewidth=2, color="#2E86AB")

    # Plot the treat-all curve.
    ax.plot(
        thresholds,
        net_benefit_all,
        label="Treat All",
        linewidth=2,
        linestyle="--",
        color="#A23B72",
    )

    # Plot the treat-none curve (net benefit = 0).
    ax.axhline(y=0, color="#F18F01", linestyle="--", linewidth=2, label="Treat None")

    # Set figure properties.
    ax.set_xlabel("Threshold Probability")
    ax.set_ylabel("Net Benefit")
    ax.set_title("Decision Curve Analysis")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])

    plt.tight_layout()
    save_figure(fig, save_path)
    plt.close()


def plot_decision_curve_analysis_v2(y_true, y_prob, save_path="LR_decision_curve.pdf"):
    """
    Plot decision curve analysis (DCA).

    Three curves:
        1. Model:      nb = TP/n - (FP/n) * (pt / (1-pt))
        2. Treat All:  nb = prevalence - (1-prevalence) * (pt / (1-pt))
        3. Treat None: nb = 0
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    n = len(y_true)
    prevalence = np.mean(y_true)

    thresholds = np.arange(0, 1.00, 0.01)
    nb_model = np.zeros(len(thresholds))
    nb_treat_all = np.zeros(len(thresholds))

    for idx, pt in enumerate(thresholds):
        if pt >= 1.0:
            nb_model[idx] = 0.0
            nb_treat_all[idx] = -np.inf
            continue

        tp = np.sum((y_prob >= pt) & (y_true == 1))
        fp = np.sum((y_prob >= pt) & (y_true == 0))
        nb_model[idx] = tp / n - (fp / n) * (pt / (1 - pt))
        nb_treat_all[idx] = prevalence - (1 - prevalence) * (pt / (1 - pt))

    nb_treat_none = np.zeros(len(thresholds))

    # ---- Plot the figure. ----
    fig, ax = plt.subplots(figsize=(5, 4))

    ax.plot(
        thresholds,
        nb_model,
        color="#E64B35",
        linewidth=2.0,
        linestyle="-",
        label="Model",
    )
    ax.plot(
        thresholds,
        nb_treat_all,
        color="#4DBBD5",
        linewidth=2.0,
        linestyle="--",
        label="Treat All",
    )
    ax.plot(
        thresholds,
        nb_treat_none,
        color="black",
        linewidth=1.5,
        linestyle="-",
        label="Treat None",
    )

    # Y-axis range.
    valid = np.isfinite(nb_treat_all)
    y_max = max(np.nanmax(nb_model), np.nanmax(nb_treat_all[valid]), 0) + 0.05
    ax.set_ylim(-0.1, y_max)
    ax.set_xlim(0, 1)

    ax.set_xlabel("Threshold Probability")
    ax.set_ylabel("Net Benefit")
    ax.set_title("Decision Curve Analysis (DCA)")

    ax.legend(loc="upper right")
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.35, linewidth=0.8)

    plt.tight_layout()
    save_figure(fig, save_path)
    print(f"Decision curve saved to: {save_path}")


def plot_calibration_curve_analysis(
    y_true, y_prob, save_path="calibration_curve.pdf", n_bins=10
):
    """
    Plot the calibration curve.

    Parameters:
        y_true: Observed outcome labels.
        y_prob: Predicted probabilities.
        save_path: Output path.
        n_bins: Number of bins.
    """
    # Calculate the calibration curve.
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="quantile"
    )

    # Calculate the Brier score.
    brier_score = brier_score_loss(y_true, y_prob)

    # Plot the figure.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left panel: Calibration curve.
    ax1.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=2,
        color="gray",
        label="Perfectly Calibrated",
        alpha=0.7,
    )
    ax1.plot(
        mean_predicted_value,
        fraction_of_positives,
        marker="o",
        linewidth=2,
        markersize=6,
        color="#E64B35",
        label="Model",
    )

    ax1.set_xlabel("Mean Predicted Probability")
    ax1.set_ylabel("Fraction of Positives")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    # Annotate the Brier score.
    ax1.text(
        0.95,
        0.05,
        f"Brier Score: {brier_score:.4f}",
        transform=ax1.transAxes,
        fontsize=11,
        verticalalignment="bottom",
        horizontalalignment="right",
    )

    # Right panel: Histogram of predicted probabilities.
    ax2.hist(
        y_prob[y_true == 0],
        bins=20,
        alpha=0.6,
        color="#4DBBD5",
        label="Negative Class",
        edgecolor="black",
    )
    ax2.hist(
        y_prob[y_true == 1],
        bins=20,
        alpha=0.6,
        color="#E64B35",
        label="Positive Class",
        edgecolor="black",
    )

    ax2.set_xlabel("Predicted Probability")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_figure(fig, save_path)
    plt.close()


def plot_clinical_impact_curve(y_true, y_prob, save_path="clinical_impact_curve.pdf"):
    """
    Plot a clinical impact curve.

    Parameters:
        y_true: Observed outcome labels.
        y_prob: Predicted probabilities.
        save_path: Output path.
    """
    # Threshold-probability range.
    thresholds = np.linspace(0.01, 0.99, 100)

    n = len(y_true)
    n_high_risk = []  # Number classified as high risk.
    n_high_risk_with_outcome = (
        []
    )  # Number of observed events among those classified as high risk.

    for threshold in thresholds:
        # Convert probabilities to predicted classes.
        y_pred = (y_prob >= threshold).astype(int)

        # Count those classified as high risk per 1,000.
        n_classified_high_risk = np.sum(y_pred) / n * 1000
        n_high_risk.append(n_classified_high_risk)

        # Count true positives among those classified as high risk per 1,000.
        TP = np.sum((y_pred == 1) & (y_true == 1))
        n_tp = TP / n * 1000
        n_high_risk_with_outcome.append(n_tp)

    # Plot the figure.
    fig, ax = plt.subplots(figsize=(5, 4))

    # Plot the number classified as high risk.
    ax.plot(
        thresholds,
        n_high_risk,
        label="Number High Risk",
        linewidth=2.5,
        color="#E63946",
        linestyle="-",
    )

    # Plot true positives among those classified as high risk.
    ax.plot(
        thresholds,
        n_high_risk_with_outcome,
        label="Number High Risk with Outcome",
        linewidth=2.5,
        color="#2A9D8F",
        linestyle="--",
    )

    # Set figure properties.
    ax.set_xlabel("Threshold Probability")
    ax.set_ylabel("Number per 1000 Visits")
    ax.set_title("Clinical Impact Curve")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, max(n_high_risk) * 1.1])

    # Add explanatory text.
    prevalence = np.sum(y_true) / n
    ax.text(
        0.95,
        0.07,
        f"Prevalence: {prevalence:.2%}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    save_figure(fig, save_path)
    plt.close()


def plot_combined_analysis(
    y_true, y_prob, save_path="combined_analysis.pdf", n_bins=10
):
    """
    Combine calibration, probability distributions, DCA, and CIC in a 2x2 grid.
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    n = len(y_true)
    prevalence = np.mean(y_true)

    fig, axes = fs.axes_grid(
        2, 2, hgap=20.0
    )  # Increase row spacing to separate upper-panel x labels from lower-panel titles.
    ax_cal, ax_hist, ax_dca, ax_cic = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # ── Upper left: Calibration curve. ──────────────────────────────────────────────
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="quantile"
    )
    brier_score = brier_score_loss(y_true, y_prob)

    ax_cal.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.3,
        color="gray",
        label="Perfectly Calibrated",
        alpha=0.7,
    )
    ax_cal.plot(
        mean_predicted_value,
        fraction_of_positives,
        marker="o",
        linewidth=1.3,
        markersize=4,
        color="#E64B35",
        label="Model",
    )
    ax_cal.set_xlabel("Mean Predicted Probability")
    ax_cal.set_ylabel("Fraction of Positives")
    ax_cal.set_title("Calibration Curve")
    ax_cal.legend(loc="upper left")
    ax_cal.grid(True, alpha=0.3)
    ax_cal.set_xlim([0, 1])
    ax_cal.set_ylim([0, 1])
    ax_cal.text(
        0.95,
        0.05,
        f"Brier Score: {brier_score:.4f}",
        transform=ax_cal.transAxes,
        fontsize=fs.ANNOTATION_SIZE,
        va="bottom",
        ha="right",
    )

    # ── Upper right: Predicted-probability distribution. ──────────────────────────────────────────
    ax_hist.hist(
        y_prob[y_true == 0],
        bins=20,
        alpha=0.6,
        color="#4DBBD5",
        label="Negative Class",
        edgecolor="black",
    )
    ax_hist.hist(
        y_prob[y_true == 1],
        bins=20,
        alpha=0.6,
        color="#E64B35",
        label="Positive Class",
        edgecolor="black",
    )
    ax_hist.set_xlabel("Predicted Probability")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("Predicted Probability Distribution")
    ax_hist.legend(loc="upper right")
    ax_hist.grid(True, alpha=0.3, axis="y")

    # ── Lower left: DCA curve. ──────────────────────────────────────────────
    thresholds = np.arange(0, 1.00, 0.01)
    nb_model = np.zeros(len(thresholds))
    nb_treat_all = np.zeros(len(thresholds))

    for idx, pt in enumerate(thresholds):
        if pt >= 1.0:
            nb_model[idx] = 0.0
            nb_treat_all[idx] = -np.inf
            continue
        tp = np.sum((y_prob >= pt) & (y_true == 1))
        fp = np.sum((y_prob >= pt) & (y_true == 0))
        nb_model[idx] = tp / n - (fp / n) * (pt / (1 - pt))
        nb_treat_all[idx] = prevalence - (1 - prevalence) * (pt / (1 - pt))

    valid = np.isfinite(nb_treat_all)
    y_max = max(np.nanmax(nb_model), np.nanmax(nb_treat_all[valid]), 0) + 0.05

    ax_dca.plot(thresholds, nb_model, color="#E64B35", linewidth=1.3, label="Model")
    ax_dca.plot(
        thresholds,
        nb_treat_all,
        color="#0088AA",
        linewidth=1.3,
        linestyle="--",
        label="Treat All",
    )
    ax_dca.plot(
        thresholds,
        np.zeros_like(thresholds),
        color="black",
        linewidth=1.0,
        label="Treat None",
    )
    ax_dca.set_ylim(-0.1, y_max)
    ax_dca.set_xlim(0, 1)
    ax_dca.set_xlabel("Threshold Probability")
    ax_dca.set_ylabel("Net Benefit")
    ax_dca.set_title("Decision Curve Analysis")
    ax_dca.legend(loc="upper right")
    ax_dca.grid(True, linestyle="--", alpha=0.35, linewidth=0.8)

    # ── Lower right: CIC curve. ──────────────────────────────────────────────
    thresholds_cic = np.linspace(0.01, 0.99, 100)
    n_high_risk, n_high_risk_with_outcome = [], []

    for pt in thresholds_cic:
        y_pred = (y_prob >= pt).astype(int)
        n_high_risk.append(np.sum(y_pred) / n * 1000)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        n_high_risk_with_outcome.append(tp / n * 1000)

    ax_cic.plot(
        thresholds_cic,
        n_high_risk,
        label="Number High Risk",
        linewidth=1.3,
        color="#E63946",
    )
    ax_cic.plot(
        thresholds_cic,
        n_high_risk_with_outcome,
        label="Number High Risk with Outcome",
        linewidth=1.3,
        color="#0088AA",
        linestyle="--",
    )
    ax_cic.set_xlabel("Threshold Probability")
    ax_cic.set_ylabel("Number per 1000 Visits")
    ax_cic.set_title("Clinical Impact Curve")
    ax_cic.legend(loc="upper right")
    ax_cic.grid(True, alpha=0.3)
    ax_cic.set_xlim([0, 1])
    ax_cic.set_ylim([0, max(n_high_risk) * 1.1])
    ax_cic.text(
        0.95,
        0.05,
        f"Prevalence: {prevalence:.2%}",
        transform=ax_cic.transAxes,
        fontsize=fs.ANNOTATION_SIZE,
        va="bottom",
        ha="right",
    )

    # Add consistent panel letters.
    for ax, label in zip(axes.flat, ["A", "B", "C", "D"]):
        fs.panel_letter(ax, label)

    save_figure(fig, save_path)
    plt.close()
    print(f"Combined analysis saved to: {save_path}")
