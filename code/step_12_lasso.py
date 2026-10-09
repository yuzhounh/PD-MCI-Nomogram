import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_pdf import PdfPages

# File name variables
INPUT_TRAIN_FILE = "PPMI_8_train.csv"
INPUT_TEST_FILE = "PPMI_8_test.csv"
OUTPUT_TRAIN_FILE = "PPMI_9_train.csv"
OUTPUT_TEST_FILE = "PPMI_9_test.csv"
FEATURE_MAPPING_FILE = "PPMI_feature_mapping.csv"

# Updated output filenames
COEF_PATH_FILE = "step_12_lasso_coefficients_path.csv"
LASSO_PLOT_FILE = "step_12_lasso_plot.pdf"
FEATURE_WEIGHT_CSV = "step_12_feature_weight.csv"
FEATURE_WEIGHT_PDF = "step_12_feature_weight.pdf"
CV_PLOT_DATA_FILE = "step_12_lasso_cv_plot_data.csv"

# Common figure style (fonts, sizes, axes frame)
import figstyle as fs

fs.apply()
plt.rcParams["axes.unicode_minus"] = True
# Computer Modern math font, the same as the formulas in the manuscript (the lambda labels)
plt.rcParams["mathtext.fontset"] = "cm"

# ====================== Core Functions ======================


def create_subject_proxy_labels(y, subjects):
    """Create subject-level proxy labels (if any sample is 1, subject is 1)"""
    y = np.asarray(y)
    subjects = np.asarray(subjects)
    subject_labels_map = pd.Series(y, index=subjects).groupby(level=0).max()
    y_proxy = pd.Series(subjects).map(subject_labels_map).values
    return y_proxy, subject_labels_map.to_dict()


def subject_level_cross_validation(X, y, subjects, n_folds=10, random_state=42):
    """Subject-level cross validation"""
    subject_labels_series = pd.Series(y, index=subjects).groupby(level=0).max()
    unique_subjects = subject_labels_series.index.values
    subject_labels = subject_labels_series.values

    sgkf = StratifiedGroupKFold(
        n_splits=n_folds, shuffle=True, random_state=random_state
    )
    folds = []

    fold_num = 1
    for train_subject_idx, val_subject_idx in sgkf.split(
        unique_subjects, subject_labels, groups=unique_subjects
    ):
        train_subjects_fold = unique_subjects[train_subject_idx]
        val_subjects_fold = unique_subjects[val_subject_idx]

        train_sample_idx = np.where(np.isin(subjects, train_subjects_fold))[0]
        val_sample_idx = np.where(np.isin(subjects, val_subjects_fold))[0]

        # print(f"Fold {fold_num}: Train {len(train_subjects_fold)} subjects/{len(train_sample_idx)} samples, Val {len(val_subjects_fold)} subjects/{len(val_sample_idx)} samples")
        folds.append((train_sample_idx, val_sample_idx))
        fold_num += 1

    return folds


def evaluate_model_cv(
    X, y, subjects, model, cv_folds, scoring_func=average_precision_score
):
    """Model evaluation using cross-validation with in-fold standardization to prevent data leakage"""
    scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv_folds):
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]

        # Fit the scaler on each training fold and transform its validation fold to prevent distribution leakage.
        fold_scaler = StandardScaler()
        X_train_fold_scaled = fold_scaler.fit_transform(X_train_fold)
        X_val_fold_scaled = fold_scaler.transform(X_val_fold)

        model.fit(X_train_fold_scaled, y_train_fold)

        if hasattr(model, "predict_proba"):
            y_pred = model.predict_proba(X_val_fold_scaled)[:, 1]
        else:
            y_pred = model.predict(X_val_fold_scaled)

        score = scoring_func(y_val_fold, y_pred)
        scores.append(score)

    return np.array(scores)


def compute_coefficient_path(X, y, lambda_seq, random_state=42):
    """
    Compute LASSO coefficient path on the full standardized training set.

    Method: Use 10-fold cross-validation with fold-specific scaling to determine the sparse hyperparameter
    lambda_1SE under the 1SE rule. After tuning, follow the standard two-stage approach (Hastie et al.)
    to calculate coefficient paths and select features/weights using the full standardized training set.
    """
    coefficients_path = np.zeros((len(lambda_seq), X.shape[1]))

    for i, lam in enumerate(lambda_seq):
        model = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=1.0 / lam,
            # class_weight='balanced', # OFF
            random_state=random_state,
            max_iter=1000,
        )
        model.fit(X, y)
        coefficients_path[i] = model.coef_[0]

    coefficients_df = pd.DataFrame(coefficients_path)
    coefficients_df.index = lambda_seq
    coefficients_df.to_csv(COEF_PATH_FILE, index_label="lambda")

    return coefficients_path


def get_feature_count(coeffs):
    return np.sum(np.abs(coeffs) > 1e-6)


def lasso_kfold_cv_with_coefficients(
    X, y, subjects, lambda_seq, n_folds=10, random_state=42
):
    """LASSO hyperparameter tuning using K-fold CV with 1SE Rule and in-fold standardization"""
    cv_folds = subject_level_cross_validation(X, y, subjects, n_folds, random_state)

    y_proxy, subject_labels_map = create_subject_proxy_labels(y, subjects)
    unique_subjects = np.array(list(subject_labels_map.keys()))
    subject_proxy_labels = np.array(list(subject_labels_map.values()))

    print(f"\nStarting {n_folds}-fold CV (in-fold standardization)...")

    auc_pr_scores = np.zeros((n_folds, len(lambda_seq)))

    for i, lam in enumerate(lambda_seq):
        model = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=1.0 / lam,
            # class_weight='balanced', # OFF
            random_state=random_state,
            max_iter=1000,
        )
        scores = evaluate_model_cv(
            X, y, subjects, model, cv_folds, average_precision_score
        )
        auc_pr_scores[:, i] = scores

        if (i + 1) % 10 == 0:
            print(f"Lambda progress: {(i + 1) / len(lambda_seq) * 100:.0f}%")

    mean_auc_pr = np.mean(auc_pr_scores, axis=0)
    # 1SE rule uses the standard error of the mean across folds (SD / sqrt(K),
    # consistent with cv.glmnet's cvsd.
    se_auc_pr = np.std(auc_pr_scores, axis=0) / np.sqrt(n_folds)

    # 1. Find the Best (Max Score)
    best_idx = np.argmax(mean_auc_pr)
    best_score = mean_auc_pr[best_idx]
    best_se = se_auc_pr[best_idx]

    print("\nComputing coefficient path on full standardized training set...")
    full_scaler = StandardScaler()
    X_scaled = full_scaler.fit_transform(X)
    coefficients_path = compute_coefficient_path(X_scaled, y, lambda_seq, random_state)

    # Check features at best
    feat_count_best = get_feature_count(coefficients_path[best_idx])
    print(
        f"Best AUPRC: {best_score:.4f} (λ={lambda_seq[best_idx]:.4f}), Features: {feat_count_best}"
    )

    # 2. Find 1SE Target
    # We want the simplest model (largest lambda -> higher index in our seq usually, check lambda_seq order)
    # lambda_seq was: lambda_seq = 10**np.arange(0, 3, 0.05) -> 1, ..., 1000.
    # C = 1/lambda. So larger lambda = smaller C = Stronger Regularization = Fewer features.
    # So we want the Largest Lambda (Highest Index) that satisfies the condition.

    cutoff_score = best_score - best_se

    # Find all indices where score >= cutoff
    eligible_indices = np.where(mean_auc_pr >= cutoff_score)[0]

    # We want the one with strongest regularization.
    # lambda_seq is increasing (1 -> 1000).
    # Stronger regularization = Larger Lambda = Larger Index.
    # So we take the max index from eligible ones.
    best_1se_idx = np.max(eligible_indices)

    feat_count_1se = get_feature_count(coefficients_path[best_1se_idx])
    print(f"1SE Cutoff: {cutoff_score:.4f} (Best {best_score:.4f} - SE {best_se:.4f})")
    print(
        f"1SE AUPRC: {mean_auc_pr[best_1se_idx]:.4f} (λ={lambda_seq[best_1se_idx]:.4f}), Features: {feat_count_1se}"
    )

    return mean_auc_pr, se_auc_pr, coefficients_path, best_idx, best_1se_idx


def standardize_data(train_data, test_data):
    """Data standardization"""
    subject_col = "PATNO"
    label_col = "MCI"

    feature_cols = [
        col for col in train_data.columns if col not in [subject_col, label_col]
    ]
    X_train = train_data[feature_cols]
    X_test = test_data[feature_cols]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(
        X_train_scaled, columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        X_test_scaled, columns=X_test.columns, index=X_test.index
    )

    train_scaled = pd.concat(
        [train_data[[subject_col]], X_train_scaled, train_data[[label_col]]], axis=1
    )
    test_scaled = pd.concat(
        [test_data[[subject_col]], X_test_scaled, test_data[[label_col]]], axis=1
    )

    return train_scaled, test_scaled, scaler


def plot_coefficient_paths(
    ax, lambda_seq, coefficients_path, best_lambda_idx, best_1se_idx
):
    """Draw all candidate coefficient paths without a feature-name legend."""
    for i in range(coefficients_path.shape[1]):
        ax.plot(lambda_seq, coefficients_path[:, i], linewidth=0.9, alpha=0.65)
    ax.set_xlabel("$\\lambda$", fontsize=12)
    ax.set_ylabel("Coefficient")
    ax.set_xscale("log")
    ax.set_xlim(1, 1000)
    ax.grid(True, alpha=0.3)
    ax.axvline(lambda_seq[best_lambda_idx], color="r", linestyle="--", alpha=0.4)
    ax.axvline(lambda_seq[best_1se_idx], color="g", linestyle="--", alpha=0.4)
    fs.panel_letter(ax, "B")


def plot_results(
    lambda_seq,
    mean_auc_pr_scores,
    se_auc_pr_scores,
    coefficients_path,
    feature_names,
    feature_dict,
    best_lambda_idx,
    best_1se_idx,
):
    """Plot AUPRC and coefficient path"""
    pd.DataFrame(
        {
            "lambda": lambda_seq,
            "mean_auprc": mean_auc_pr_scores,
            "se_auprc": se_auc_pr_scores,
        }
    ).to_csv(CV_PLOT_DATA_FILE, index=False)
    fig, axes_grid = fs.axes_grid(1, 2)
    ax1, ax2 = axes_grid[0]

    # AUPRC vs Lambda
    ax1.plot(
        lambda_seq, mean_auc_pr_scores, "-", linewidth=1.2, marker="o", markersize=2.5
    )
    ax1.fill_between(
        lambda_seq,
        mean_auc_pr_scores - se_auc_pr_scores,
        mean_auc_pr_scores + se_auc_pr_scores,
        alpha=0.2,
    )
    ax1.set_xlabel("$\\lambda$", fontsize=12)
    ax1.set_ylabel("AUPRC")
    ax1.set_xscale("log")
    ax1.set_xlim(1, 1000)
    ax1.grid(True, alpha=0.3)

    # Mark feature counts on top axis (optional, but finding points is enough)

    # Best Point
    best_line = ax1.axvline(
        lambda_seq[best_lambda_idx], color="r", linestyle="--", alpha=0.6
    )
    ax1.plot(
        lambda_seq[best_lambda_idx],
        mean_auc_pr_scores[best_lambda_idx],
        "ro",
        markersize=4.5,
    )

    # 1SE Point
    se_line = ax1.axvline(
        lambda_seq[best_1se_idx], color="g", linestyle="--", alpha=0.6
    )
    ax1.plot(
        lambda_seq[best_1se_idx], mean_auc_pr_scores[best_1se_idx], "go", markersize=4.5
    )

    # Match the ROC/PR table legends: rule left-aligned, AUPRC right-aligned.
    fs.table_legend(
        ax1,
        [best_line, se_line],
        ["Best", "1SE"],
        [
            (
                "AUPRC",
                [
                    f"{mean_auc_pr_scores[best_lambda_idx]:.4f}",
                    f"{mean_auc_pr_scores[best_1se_idx]:.4f}",
                ],
            )
        ],
        "Rule",
        loc="lower left",
    )
    fs.panel_letter(ax1, "A")

    # Selected predictor identities are reported separately in Figure 5.
    plot_coefficient_paths(
        ax2, lambda_seq, coefficients_path, best_lambda_idx, best_1se_idx
    )

    # Save as pdf format
    fig.savefig(LASSO_PLOT_FILE, bbox_inches="tight")

    return best_1se_idx  # Return 1SE as the chosen one for plotting weights later


def extract_selected_features(
    selected_features, train_data_original, test_data_original
):
    """Extract selected features and save (using original unscaled data)"""
    print(f"\n=== Extract Selected Features ===")

    subject_col = "PATNO"
    label_col = "MCI"

    columns_to_keep = [subject_col] + selected_features + [label_col]

    train_selected = train_data_original[columns_to_keep]
    test_selected = test_data_original[columns_to_keep]

    train_selected.to_csv(OUTPUT_TRAIN_FILE, index=False)
    test_selected.to_csv(OUTPUT_TEST_FILE, index=False)

    print(f"Saved: {OUTPUT_TRAIN_FILE}, {OUTPUT_TEST_FILE}")

    return train_selected, test_selected


def main():
    """Main function"""
    # Load data
    print("=== Load Data ===")
    train_data = pd.read_csv(INPUT_TRAIN_FILE)
    test_data = pd.read_csv(INPUT_TEST_FILE)

    # Load feature mapping
    feature_mapping = pd.read_csv(FEATURE_MAPPING_FILE)
    feature_dict = dict(
        zip(feature_mapping["feature"], feature_mapping["abbreviation"])
    )

    # Remove unnecessary columns
    columns_to_drop = ["SITE", "subgroup", "EVENT_ID", "visit_date"]
    train_data = train_data.drop(columns=columns_to_drop, errors="ignore")
    test_data = test_data.drop(columns=columns_to_drop, errors="ignore")

    # Keep original unscaled data for later saving
    train_data_original = train_data.copy()
    test_data_original = test_data.copy()

    # Standardization
    print("\n=== Standardization ===")
    train_scaled, test_scaled, scaler = standardize_data(train_data, test_data)

    # K-fold CV with in-fold standardization
    print("\n=== K-fold CV (1SE Rule, in-fold standardization) ===")
    # Extended range to ensure we find sparse solutions
    lambda_seq = 10 ** np.arange(0, 4, 0.05)

    subject_col = "PATNO"
    label_col = "MCI"
    feature_cols = [
        col for col in train_data.columns if col not in [subject_col, label_col]
    ]

    X_train_raw = train_data[feature_cols].values
    y_train = train_data[label_col].values
    subjects_train = train_data[subject_col].values

    mean_auc_pr, se_auc_pr, coefficients_path, best_lambda_idx, best_1se_idx = (
        lasso_kfold_cv_with_coefficients(
            X_train_raw, y_train, subjects_train, lambda_seq, n_folds=10
        )
    )

    # Plot results
    print("\n=== Visualization ===")
    feature_names = feature_cols
    plot_results(
        lambda_seq,
        mean_auc_pr,
        se_auc_pr,
        coefficients_path,
        feature_names,
        feature_dict,
        best_lambda_idx,
        best_1se_idx,
    )

    # Output results - WE USE 1SE INDEX NOW
    chosen_idx = best_1se_idx
    best_lambda = lambda_seq[chosen_idx]
    best_auc_pr = mean_auc_pr[chosen_idx]
    best_coeffs = coefficients_path[chosen_idx]

    print(f"\nFinal Selection (1SE Rule):")
    print(f"Lambda: {best_lambda:.6f}")
    print(f"AUPRC: {best_auc_pr:.4f}")

    # Select non-zero features
    non_zero_mask = np.abs(best_coeffs) > 1e-6
    selected_features = [
        feature_names[i] for i in range(len(feature_names)) if non_zero_mask[i]
    ]
    selected_weights = [
        best_coeffs[i] for i in range(len(best_coeffs)) if non_zero_mask[i]
    ]

    print(f"Selected features: {len(selected_features)}/{len(feature_names)}")

    if len(selected_features) == 0:
        print("WARNING: No features selected! Falling back to 'Best' lambda.")
        chosen_idx = best_lambda_idx
        best_coeffs = coefficients_path[chosen_idx]
        non_zero_mask = np.abs(best_coeffs) > 1e-6
        selected_features = [
            feature_names[i] for i in range(len(feature_names)) if non_zero_mask[i]
        ]
        selected_weights = [
            best_coeffs[i] for i in range(len(best_coeffs)) if non_zero_mask[i]
        ]
        print(f"Fallback features: {len(selected_features)}")

    # Save feature weights
    results_df = pd.DataFrame(
        {
            "Feature": selected_features,
            "Abbreviation": [feature_dict.get(f, f) for f in selected_features],
            "Weight": selected_weights,
        }
    )
    results_df = results_df.sort_values("Weight", key=abs, ascending=False)
    results_df.to_csv(FEATURE_WEIGHT_CSV, index=False)
    print("\nSelected features:")
    print(results_df.to_string(index=False))
    # Plot feature weight bar chart (Nature style)
    fig, axes_grid = fs.axes_grid(1, 1, left=20.0, right=3.0)
    ax = axes_grid[0, 0]
    results_sorted = results_df.copy().sort_values("Weight", key=abs, ascending=True)
    alphas = [0.95 if w > 0 else 0.75 for w in results_sorted["Weight"]]
    colors = ["#3CADC4" if w > 0 else "#E64B35" for w in results_sorted["Weight"]]

    # Plot bars with individual alpha values
    for i, (weight, color, alpha) in enumerate(
        zip(results_sorted["Weight"], colors, alphas)
    ):
        ax.barh(i, abs(weight), color=color, alpha=alpha)

    ax.set_yticks(range(len(results_sorted)))
    ax.set_yticklabels(results_sorted["Abbreviation"])
    ax.set_xlabel("Weight")

    # Adjust xlim
    max_w = results_sorted["Weight"].abs().max() if not results_sorted.empty else 0.1
    ax.set_xlim(0, max_w * 1.3)

    for i, weight in enumerate(results_sorted["Weight"]):
        ax.text(
            abs(weight) + 0.005,
            i,
            f"{weight:.4f}",
            ha="left",
            va="center",
            fontsize=fs.ANNOTATION_SIZE,
        )

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#3CADC4", alpha=0.95, label="Positive"),
        Patch(facecolor="#E64B35", alpha=0.75, label="Negative"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    ax.grid(axis="x", alpha=0.3, linestyle="--")
    fig.savefig(FEATURE_WEIGHT_PDF, bbox_inches="tight")

    # Extract selected features (using original unscaled data)
    extract_selected_features(
        selected_features, train_data_original, test_data_original
    )

    print("\n=== Complete ===")
    print(f"Output files:")
    print(f"- {FEATURE_WEIGHT_CSV}")
    print(f"- {FEATURE_WEIGHT_PDF}")
    print(f"- {LASSO_PLOT_FILE}")


if __name__ == "__main__":
    main()
