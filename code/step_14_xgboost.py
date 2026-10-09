import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import average_precision_score
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


class XGBoostModel:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.optimal_threshold = None
        self.trial_count = 0

    def validate_params(self, params):
        """Validate parameter compatibility."""
        return True

    def create_model(self, params):
        """Create a model from the parameters."""
        (
            n_estimators,
            learning_rate,
            max_depth,
            subsample,
            colsample_bytree,
            reg_alpha,
            reg_lambda,
            gamma,
            min_child_weight,
            scale_pos_weight,
        ) = params

        if not self.validate_params(params):
            return None

        model_params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "gamma": gamma,
            "min_child_weight": min_child_weight,
            "scale_pos_weight": scale_pos_weight,
            "random_state": self.random_state,
            "n_jobs": -1,
            "eval_metric": "logloss",
            "verbose": False,
        }

        return xgb.XGBClassifier(**model_params)

    def objective_function(self, trial, X, y, subjects, folds, study):
        """Objective function for Optuna Bayesian optimization."""
        self.trial_count += 1

        try:
            n_estimators = trial.suggest_int("n_estimators", 50, 500)
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
            max_depth = trial.suggest_int("max_depth", 2, 6)
            subsample = trial.suggest_float("subsample", 0.6, 1.0)
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)
            reg_alpha = trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True)
            reg_lambda = trial.suggest_float("reg_lambda", 1.0, 10.0, log=True)
            gamma = trial.suggest_float("gamma", 0.0, 0.5)
            min_child_weight = trial.suggest_int("min_child_weight", 1, 20)
            scale_pos_weight = trial.suggest_float("scale_pos_weight", 1.0, 5.0)

            params = [
                n_estimators,
                learning_rate,
                max_depth,
                subsample,
                colsample_bytree,
                reg_alpha,
                reg_lambda,
                gamma,
                min_child_weight,
                scale_pos_weight,
            ]

            model = self.create_model(params)

            if model is None:
                return 0.0

            auc_scores = []
            for train_idx, val_idx in folds:
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]

                model.fit(
                    X_train_fold,
                    y_train_fold,
                    eval_set=[(X_val_fold, y_val_fold)],
                    verbose=False,
                )

                y_val_prob = model.predict_proba(X_val_fold)[:, 1]
                auc_pr = average_precision_score(y_val_fold, y_val_prob)
                auc_scores.append(auc_pr)

            mean_auc = np.mean(auc_scores)
            return mean_auc

        except Exception as e:
            print(f"Error in XGBoost objective function: {e}")
            return 0.0

    def optimize_hyperparameters(self, X, y, subjects, n_calls=100):
        """Tune hyperparameters with Optuna Bayesian optimization."""
        print("Starting XGBoost Bayesian optimization...")
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
            params_list.append(f"n_estimators={trial.params['n_estimators']}")
            params_list.append(f"learning_rate={trial.params['learning_rate']:.4f}")
            params_list.append(f"max_depth={trial.params['max_depth']}")
            params_list.append(f"subsample={trial.params['subsample']:.4f}")
            params_list.append(
                f"colsample_bytree={trial.params['colsample_bytree']:.4f}"
            )
            params_list.append(f"reg_alpha={trial.params['reg_alpha']:.4f}")
            params_list.append(f"reg_lambda={trial.params['reg_lambda']:.4f}")
            params_list.append(f"gamma={trial.params['gamma']:.4f}")
            params_list.append(f"min_child_weight={trial.params['min_child_weight']}")
            params_list.append(
                f"scale_pos_weight={trial.params['scale_pos_weight']:.4f}"
            )

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

    def train_final_model(self, X, y, subjects):
        """Train the final model on the complete training set."""
        print("Training final XGBoost model...")

        self.model = self.create_model(
            [
                self.best_params["n_estimators"],
                self.best_params["learning_rate"],
                self.best_params["max_depth"],
                self.best_params["subsample"],
                self.best_params["colsample_bytree"],
                self.best_params["reg_alpha"],
                self.best_params["reg_lambda"],
                self.best_params["gamma"],
                self.best_params["min_child_weight"],
                self.best_params["scale_pos_weight"],
            ]
        )

        if self.model is None:
            raise ValueError("Model creation failed. Please check the parameters.")

        self.model.fit(X, y, verbose=False)

        # Calculate optimal thresholds.
        folds = subject_level_cross_validation(
            X, y, subjects, n_folds=10, random_state=self.random_state
        )
        youden_thresholds = []
        f1_thresholds = []

        for train_idx, val_idx in folds:
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            fold_model = self.create_model(
                [
                    self.best_params["n_estimators"],
                    self.best_params["learning_rate"],
                    self.best_params["max_depth"],
                    self.best_params["subsample"],
                    self.best_params["colsample_bytree"],
                    self.best_params["reg_alpha"],
                    self.best_params["reg_lambda"],
                    self.best_params["gamma"],
                    self.best_params["min_child_weight"],
                    self.best_params["scale_pos_weight"],
                ]
            )

            fold_model.fit(
                X_train_fold,
                y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                verbose=False,
            )

            y_val_prob = fold_model.predict_proba(X_val_fold)[:, 1]

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
        """Get cross-validation performance."""
        print("Calculating training set cross-validation performance...")

        folds = subject_level_cross_validation(
            X, y, subjects, n_folds=10, random_state=self.random_state
        )

        fold_results = {"default": [], "f1": [], "youden": []}

        youden_thresholds = []
        f1_thresholds = []

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            fold_model = self.create_model(
                [
                    self.best_params["n_estimators"],
                    self.best_params["learning_rate"],
                    self.best_params["max_depth"],
                    self.best_params["subsample"],
                    self.best_params["colsample_bytree"],
                    self.best_params["reg_alpha"],
                    self.best_params["reg_lambda"],
                    self.best_params["gamma"],
                    self.best_params["min_child_weight"],
                    self.best_params["scale_pos_weight"],
                ]
            )

            fold_model.fit(
                X_train_fold,
                y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                verbose=False,
            )

            y_val_prob = fold_model.predict_proba(X_val_fold)[:, 1]

            f1_thresh = f1_score_threshold(y_val_fold, y_val_prob)
            youden_thresh = youden_threshold(y_val_fold, y_val_prob)

            f1_thresholds.append(f1_thresh)
            youden_thresholds.append(youden_thresh)

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
        """Evaluate feature importance."""
        print("Calculating XGBoost feature importance...")

        sample_size = min(500, len(X))
        sample_indices = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[sample_indices]

        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_sample)

        plot_shap_summary(shap_values, X_sample, feature_names, "XGBoost", save_path)

        return shap_values

    def evaluate_and_save_results(
        self, X_test, y_test, save_path="xgboost_predictions.csv"
    ):
        """Evaluate test-set performance."""
        print("Calculating test set performance...")
        y_prob = self.model.predict_proba(X_test)[:, 1]

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

    def plot_decision_curve(
        self, y_true, y_prob, save_path="xgboost_decision_curve.pdf"
    ):
        """Plot the decision curve."""
        print("Plotting decision curve...")
        plot_decision_curve_analysis_v2(y_true, y_prob, save_path)
        print(f"Decision curve saved to: {save_path}")

    def plot_calibration_curve(
        self, y_true, y_prob, save_path="xgboost_calibration_curve.pdf"
    ):
        """Plot the calibration curve."""
        print("Plotting calibration curve...")
        plot_calibration_curve_analysis(y_true, y_prob, save_path)
        print(f"Calibration curve saved to: {save_path}")

    def plot_clinical_impact_curve(
        self, y_true, y_prob, save_path="xgboost_clinical_impact_curve.pdf"
    ):
        """Plot the clinical impact curve."""
        print("Plotting clinical impact curve...")
        plot_clinical_impact_curve(y_true, y_prob, save_path)
        print(f"Clinical impact curve saved to: {save_path}")

    def plot_roc_pr_curves(self, y_true, y_prob, save_path="xgboost_roc_pr_curves.pdf"):
        """Plot ROC and PR curves."""
        print("Plotting ROC and PR curves...")
        plot_roc_pr_curves(y_true, y_prob, save_path, model_name="XGBoost")
        print(f"ROC and PR curves saved to: {save_path}")

    def plot_combined_analysis(
        self, y_true, y_prob, save_path="xgboost_combined_analysis.pdf"
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

    # Create the XGBoost model.
    xgb_model = XGBoostModel(random_state=42)

    # Optimize hyperparameters.
    xgb_model.optimize_hyperparameters(X_train, y_train, subjects_train)

    # Train the final model.
    xgb_model.train_final_model(X_train, y_train, subjects_train)

    # Get cross-validation performance.
    train_performance = xgb_model.get_cross_validation_performance(
        X_train, y_train, subjects_train
    )

    # Evaluate feature importance.
    xgb_model.evaluate_feature_importance(
        X_train, y_train, feature_names, "step_14_xgboost_shap"
    )

    # Evaluate test-set performance.
    test_performance, y_prob = xgb_model.evaluate_and_save_results(
        X_test, y_test, "step_14_xgboost_predictions.csv"
    )

    # Plot ROC and PR curves.
    xgb_model.plot_roc_pr_curves(y_test, y_prob, "step_14_xgboost_roc_pr_curves.pdf")

    # # Plot the decision curve.
    # xgb_model.plot_decision_curve(y_test, y_prob, 'step_14_xgboost_decision_curve.pdf')

    # # Plot the calibration curve.
    # xgb_model.plot_calibration_curve(y_test, y_prob, 'step_14_xgboost_calibration_curve.pdf')

    # # Plot the clinical impact curve.
    # xgb_model.plot_clinical_impact_curve(y_test, y_prob, 'step_14_xgboost_clinical_impact_curve.pdf')

    # Plot the combined figure.
    xgb_model.plot_combined_analysis(
        y_test, y_prob, "step_14_xgboost_combined_analysis.pdf"
    )

    # Save performance results.
    train_df = pd.DataFrame(train_performance)
    train_df.to_csv("step_14_xgboost_train_performance.csv")
    print(f"\nTraining set performance saved to: step_14_xgboost_train_performance.csv")

    test_df = pd.DataFrame(test_performance)
    test_df.to_csv("step_14_xgboost_test_performance.csv")
    print(f"Test set performance saved to: step_14_xgboost_test_performance.csv")

    # Print the results.
    print("\n" + "=" * 80)
    print("XGBoost model training set performance (mean±std):")
    print("=" * 80)
    print(train_df.round(4))

    print("\n" + "=" * 80)
    print("XGBoost model test set performance:")
    print("=" * 80)
    print(test_df.round(4))


if __name__ == "__main__":
    main()
