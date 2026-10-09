import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
import optuna
import shap
from utils import *
import warnings

warnings.filterwarnings("ignore")

# Use Arial fonts.
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = True
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Arial"
plt.rcParams["mathtext.it"] = "Arial:italic"
plt.rcParams["mathtext.bf"] = "Arial:bold"


class MLPModel:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.best_params = None
        self.optimal_threshold = None
        self.trial_count = 0

    def validate_params(self, params):
        """Validate parameter compatibility."""
        return True

    def create_model(self, params):
        """Create a model from the parameters."""
        (
            n_layers,
            n_units,
            alpha,
            learning_rate_init,
            batch_size,
            max_iter,
        ) = params

        if not self.validate_params(params):
            return None

        hidden_layer_sizes = tuple([n_units] * n_layers)

        model_params = {
            "hidden_layer_sizes": hidden_layer_sizes,
            "activation": "relu",
            "solver": "adam",
            "alpha": alpha,
            "learning_rate_init": learning_rate_init,
            "batch_size": batch_size,
            "max_iter": max_iter,
            "early_stopping": True,
            "n_iter_no_change": 10,
            "random_state": self.random_state,
        }

        return MLPClassifier(**model_params)

    def objective_function(self, trial, X, y, subjects, folds, study):
        """Objective function for Optuna Bayesian optimization."""
        self.trial_count += 1

        try:
            n_layers = trial.suggest_int("n_layers", 1, 3)
            n_units = trial.suggest_int("n_units", 8, 128)
            alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
            learning_rate_init = trial.suggest_float(
                "learning_rate_init", 1e-4, 1e-1, log=True
            )
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
            max_iter = trial.suggest_int("max_iter", 100, 500)

            params = [
                n_layers,
                n_units,
                alpha,
                learning_rate_init,
                batch_size,
                max_iter,
            ]

            model = self.create_model(params)

            if model is None:
                return 0.0

            auc_scores = []
            for train_idx, val_idx in folds:
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]

                scaler = StandardScaler()
                X_train_fold_s = scaler.fit_transform(X_train_fold)
                X_val_fold_s = scaler.transform(X_val_fold)

                model.fit(X_train_fold_s, y_train_fold)

                y_val_prob = model.predict_proba(X_val_fold_s)[:, 1]
                auc_pr = average_precision_score(y_val_fold, y_val_prob)
                auc_scores.append(auc_pr)

            mean_auc = np.mean(auc_scores)
            return mean_auc

        except Exception as e:
            print(f"Error in MLP objective function: {e}")
            return 0.0

    def optimize_hyperparameters(self, X, y, subjects, n_calls=100):
        """Tune hyperparameters with Optuna Bayesian optimization."""
        print("Starting MLP Bayesian optimization...")
        print("=" * 80)

        folds = subject_level_cross_validation(
            X, y, subjects, n_folds=10, random_state=self.random_state
        )

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )

        self.trial_count = 0

        def callback(study, trial):
            if trial.state != optuna.trial.TrialState.COMPLETE or not trial.params:
                return

            current_trial = self.trial_count
            total_trials = n_calls

            params_list = []
            params_list.append(f"n_layers={trial.params['n_layers']}")
            params_list.append(f"n_units={trial.params['n_units']}")
            params_list.append(f"alpha={trial.params['alpha']:.6f}")
            params_list.append(
                f"learning_rate_init={trial.params['learning_rate_init']:.6f}"
            )
            params_list.append(f"batch_size={trial.params['batch_size']}")
            params_list.append(f"max_iter={trial.params['max_iter']}")

            params_str = ", ".join(params_list)
            current_auc_pr = trial.value
            best_auc_pr = study.best_value
            best_trial_number = study.best_trial.number + 1

            print(f"Trial: {current_trial}/{total_trials}")
            print(f"Parameters: {params_str}")
            print(f"AUC-PR: {current_auc_pr:.6f}")
            print(f"Best AUC-PR (Trial {best_trial_number}): {best_auc_pr:.6f}")
            print()

        study.optimize(
            lambda trial: self.objective_function(trial, X, y, subjects, folds, study),
            n_trials=n_calls,
            callbacks=[callback],
        )

        print("=" * 80)
        print("Search completed")

        best_trial = study.best_trial
        self.best_params = best_trial.params.copy()

        print(f"Best parameters: {self.best_params}")
        print(
            f"Best objective value: {best_trial.value:.6f} (Best AUC-PR: {best_trial.value:.6f})"
        )

        return study

    def _params_list(self):
        """Convert best_params to a list."""
        return [
            self.best_params["n_layers"],
            self.best_params["n_units"],
            self.best_params["alpha"],
            self.best_params["learning_rate_init"],
            self.best_params["batch_size"],
            self.best_params["max_iter"],
        ]

    def train_final_model(self, X, y, subjects):
        """Train the final model on the complete training set."""
        print("Training final MLP model...")

        self.model = self.create_model(self._params_list())

        if self.model is None:
            raise ValueError("Model creation failed. Please check the parameters.")

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

        # Calculate optimal thresholds using fold-specific scaling.
        folds = subject_level_cross_validation(
            X, y, subjects, n_folds=10, random_state=self.random_state
        )
        youden_thresholds = []
        f1_thresholds = []

        for train_idx, val_idx in folds:
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            fold_scaler = StandardScaler()
            X_train_fold_s = fold_scaler.fit_transform(X_train_fold)
            X_val_fold_s = fold_scaler.transform(X_val_fold)

            fold_model = self.create_model(self._params_list())
            fold_model.fit(X_train_fold_s, y_train_fold)

            y_val_prob = fold_model.predict_proba(X_val_fold_s)[:, 1]

            youden_thresh = youden_threshold(y_val_fold, y_val_prob)
            f1_thresh = f1_score_threshold(y_val_fold, y_val_prob)

            youden_thresholds.append(youden_thresh)
            f1_thresholds.append(f1_thresh)

        self.default = 0.5
        self.f1_median = np.median(f1_thresholds)
        self.youden_median = np.median(youden_thresholds)

        print("\n" + "=" * 60)
        print("Optimal threshold calculation results:")
        print("=" * 60)
        print(f"1. Default threshold:   0.5000")
        print(f"2. F1-score:           {self.f1_median:.4f}")
        print(f"3. Youden index:       {self.youden_median:.4f}")
        print("=" * 60)

        print("\nThreshold details by fold:")
        print("Fold\tYouden\tF1-score")
        print("-" * 35)
        for i, (y_thresh, f_thresh) in enumerate(
            zip(youden_thresholds, f1_thresholds), 1
        ):
            print(f"{i:2d}\t{y_thresh:.4f}\t{f_thresh:.4f}")
        print("-" * 35)
        print(f"Median\t{self.youden_median:.4f}\t{self.f1_median:.4f}")
        print()

    def get_cross_validation_performance(self, X, y, subjects):
        """Get cross-validation performance using fold-specific scaling."""
        print("Calculating training set cross-validation performance...")

        folds = subject_level_cross_validation(
            X, y, subjects, n_folds=10, random_state=self.random_state
        )

        fold_results = {"default": [], "f1": [], "youden": []}

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            fold_scaler = StandardScaler()
            X_train_fold_s = fold_scaler.fit_transform(X_train_fold)
            X_val_fold_s = fold_scaler.transform(X_val_fold)

            fold_model = self.create_model(self._params_list())
            fold_model.fit(X_train_fold_s, y_train_fold)

            y_val_prob = fold_model.predict_proba(X_val_fold_s)[:, 1]

            f1_thresh = f1_score_threshold(y_val_fold, y_val_prob)
            youden_thresh = youden_threshold(y_val_fold, y_val_prob)

            thresholds = {"default": 0.5, "f1": f1_thresh, "youden": youden_thresh}

            for thresh_type, threshold in thresholds.items():
                y_pred = (y_val_prob >= threshold).astype(int)
                metrics = calculate_all_metrics(
                    y_val_fold, y_pred, y_val_prob, threshold
                )
                fold_results[thresh_type].append(metrics)

        summary_results = {}

        for thresh_type in ["default", "f1", "youden"]:
            metrics_df = pd.DataFrame(fold_results[thresh_type])
            means = metrics_df.mean()
            stds = metrics_df.std()

            summary_results[f"train_{thresh_type}_mean"] = means.to_dict()
            summary_results[f"train_{thresh_type}_std"] = stds.to_dict()

        return summary_results

    def evaluate_feature_importance(self, X, y, feature_names, save_path):
        """Evaluate feature importance with KernelExplainer because MLP is not a tree model."""
        print("Calculating MLP feature importance...")

        background_size = min(100, len(X))
        sample_size = min(500, len(X))

        background_indices = np.random.choice(len(X), background_size, replace=False)
        sample_indices = np.random.choice(len(X), sample_size, replace=False)

        X_background = X[background_indices]
        X_sample = X[sample_indices]

        if hasattr(self, "scaler") and self.scaler is not None:
            X_background_scaled = self.scaler.transform(X_background)
            X_sample_scaled = self.scaler.transform(X_sample)
        else:
            X_background_scaled = X_background
            X_sample_scaled = X_sample

        explainer = shap.KernelExplainer(self.model.predict_proba, X_background_scaled)
        shap_values = explainer.shap_values(X_sample_scaled, nsamples=100)

        # Select positive-class SHAP values for binary classification.
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        plot_shap_summary(shap_values, X_sample_scaled, feature_names, "MLP", save_path)

        return shap_values

    def evaluate_and_save_results(
        self, X_test, y_test, save_path="mlp_predictions.csv"
    ):
        """Evaluate test-set performance."""
        print("Calculating test set performance...")
        if hasattr(self, "scaler") and self.scaler is not None:
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_test_scaled = X_test
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]

        test_performance = {}

        # Default threshold.
        y_pred_default = (y_prob >= self.default).astype(int)
        test_performance["test_default"] = calculate_all_metrics(
            y_test, y_pred_default, y_prob, self.default
        )

        # F1 threshold.
        y_pred_f1 = (y_prob >= self.f1_median).astype(int)
        test_performance["test_f1"] = calculate_all_metrics(
            y_test, y_pred_f1, y_prob, self.f1_median
        )

        # Youden threshold.
        y_pred_youden = (y_prob >= self.youden_median).astype(int)
        test_performance["test_youden"] = calculate_all_metrics(
            y_test, y_pred_youden, y_prob, self.youden_median
        )

        # Save predictions.
        predictions_data = pd.DataFrame({"y_true": y_test, "y_prob": y_prob})
        predictions_data.to_csv(save_path, index=False)
        print(f"Predictions saved to: {save_path}")

        return test_performance, y_prob

    def plot_decision_curve(self, y_true, y_prob, save_path="mlp_decision_curve.pdf"):
        """Plot the decision curve."""
        print("Plotting decision curve...")
        plot_decision_curve_analysis_v2(y_true, y_prob, save_path)
        print(f"Decision curve saved to: {save_path}")

    def plot_calibration_curve(
        self, y_true, y_prob, save_path="mlp_calibration_curve.pdf"
    ):
        """Plot the calibration curve."""
        print("Plotting calibration curve...")
        plot_calibration_curve_analysis(y_true, y_prob, save_path)
        print(f"Calibration curve saved to: {save_path}")

    def plot_clinical_impact_curve(
        self, y_true, y_prob, save_path="mlp_clinical_impact_curve.pdf"
    ):
        """Plot the clinical impact curve."""
        print("Plotting clinical impact curve...")
        plot_clinical_impact_curve(y_true, y_prob, save_path)
        print(f"Clinical impact curve saved to: {save_path}")

    def plot_roc_pr_curves(self, y_true, y_prob, save_path="mlp_roc_pr_curves.pdf"):
        """Plot ROC and PR curves."""
        print("Plotting ROC and PR curves...")
        plot_roc_pr_curves(y_true, y_prob, save_path, model_name="MLP")
        print(f"ROC and PR curves saved to: {save_path}")

    def plot_combined_analysis(
        self, y_true, y_prob, save_path="mlp_combined_analysis.pdf"
    ):
        """Plot calibration, probability distributions, DCA, and CIC in a combined figure."""
        print("Plotting combined analysis...")
        plot_combined_analysis(y_true, y_prob, save_path)
        print(f"Combined analysis saved to: {save_path}")


def main():
    # Set the random seed.
    np.random.seed(42)

    # Load the data.
    train_data, test_data, feature_name_mapping = load_data(
        "PPMI_9_train.csv", "PPMI_9_test.csv", "PPMI_feature_mapping.csv"
    )

    # Prepare the data.
    X_train, y_train, subjects_train = prepare_data(train_data)
    X_test, y_test, subjects_test = prepare_data(test_data)

    # Convert to NumPy arrays.
    X_train = X_train.values
    y_train = y_train.values
    subjects_train = subjects_train.values
    X_test = X_test.values
    y_test = y_test.values

    # Get feature names.
    feature_names = [
        feature_name_mapping.get(col, col)
        for col in train_data.columns
        if col not in ["PATNO", "MCI"]
    ]

    # Create the MLP model.
    mlp_model = MLPModel(random_state=42)

    # Optimize hyperparameters.
    mlp_model.optimize_hyperparameters(X_train, y_train, subjects_train)

    # Train the final model.
    mlp_model.train_final_model(X_train, y_train, subjects_train)

    # Get cross-validation performance.
    train_performance = mlp_model.get_cross_validation_performance(
        X_train, y_train, subjects_train
    )

    # Evaluate feature importance.
    mlp_model.evaluate_feature_importance(
        X_train, y_train, feature_names, "step_14_mlp_shap"
    )

    # Evaluate test-set performance.
    test_performance, y_prob = mlp_model.evaluate_and_save_results(
        X_test, y_test, "step_14_mlp_predictions.csv"
    )

    # Plot ROC and PR curves.
    mlp_model.plot_roc_pr_curves(y_test, y_prob, "step_14_mlp_roc_pr_curves.pdf")

    # # Plot the decision curve.
    # mlp_model.plot_decision_curve(y_test, y_prob, 'step_14_mlp_decision_curve.pdf')

    # # Plot the calibration curve.
    # mlp_model.plot_calibration_curve(y_test, y_prob, 'step_14_mlp_calibration_curve.pdf')

    # # Plot the clinical impact curve.
    # mlp_model.plot_clinical_impact_curve(y_test, y_prob, 'step_14_mlp_clinical_impact_curve.pdf')

    # Plot the combined figure.
    mlp_model.plot_combined_analysis(
        y_test, y_prob, "step_14_mlp_combined_analysis.pdf"
    )

    # Save performance results.
    train_df = pd.DataFrame(train_performance)
    train_df.to_csv("step_14_mlp_train_performance.csv")
    print(f"\nTraining set performance saved to: step_14_mlp_train_performance.csv")

    test_df = pd.DataFrame(test_performance)
    test_df.to_csv("step_14_mlp_test_performance.csv")
    print(f"Test set performance saved to: step_14_mlp_test_performance.csv")

    # Print the results.
    print("\n" + "=" * 80)
    print("MLP model training set performance (mean±std):")
    print("=" * 80)
    print(train_df.round(4))

    print("\n" + "=" * 80)
    print("MLP model test set performance:")
    print("=" * 80)
    print(test_df.round(4))


if __name__ == "__main__":
    main()
