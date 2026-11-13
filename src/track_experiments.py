from pathlib import Path
import json
import warnings

import joblib
import mlflow
import mlflow.sklearn


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "optimised_churn_model.joblib"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# --------------------------------------------------
# 2. Experiment configuration
# --------------------------------------------------

EXPERIMENT_NAME = "ChurnOps-Customer-Churn"

# SQLite database for MLflow tracking
MLFLOW_DATABASE = PROJECT_ROOT / "mlflow.db"

mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DATABASE.as_posix()}"
)

mlflow.set_experiment(EXPERIMENT_NAME)


# --------------------------------------------------
# 3. Load the existing trained model
# --------------------------------------------------

print("Loading the optimised churn prediction model...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}\n"
        "Check that optimised_churn_model.joblib exists in models/."
    )

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")


# --------------------------------------------------
# 4. Record model parameters and test metrics
# --------------------------------------------------

model_parameters = {
    "algorithm": "Logistic Regression",
    "C": 0.1,
    "class_weight": "balanced",
    "solver": "lbfgs",
    "max_iter": 3000,
    "random_state": 42,
    "test_size": 0.2,
    "cross_validation_folds": 5,
    "classification_threshold": 0.54,
}

# Metrics from the untouched test set, using threshold 0.54
test_metrics = {
    "accuracy": 0.7544,
    "precision": 0.5260,
    "recall": 0.7567,
    "f1_score": 0.6206,
    "roc_auc": 0.8410,
    "pr_auc": 0.6333,
}

threshold_configuration = {
    "default_threshold": 0.50,
    "optimised_threshold": 0.54,
    "threshold_selection_method": (
        "Maximum F1-score using out-of-fold training probabilities"
    ),
    "validation_f1_score": 0.6374,
    "test_evaluation": "Untouched stratified test set",
}


# --------------------------------------------------
# 5. Start an MLflow experiment run
# --------------------------------------------------

print("\nStarting MLflow experiment tracking...")

with mlflow.start_run(run_name="Optimised-Logistic-Regression"):

    # Record parameters
    mlflow.log_params(model_parameters)

    # Record evaluation metrics
    mlflow.log_metrics(test_metrics)

    # Record descriptive tags
    mlflow.set_tags({
        "project": "ChurnOps",
        "task": "Customer Churn Prediction",
        "model_type": "Classification",
        "model_status": "Baseline Optimised Model",
        "explainability": "SHAP",
        "threshold_optimisation": "Enabled",
    })

    # Record threshold configuration
    mlflow.log_dict(
        threshold_configuration,
        "configuration/threshold_configuration.json"
    )

    # Record the trained Scikit-learn pipeline
    mlflow.sklearn.log_model(
        sk_model=model,
        name="churn_prediction_model"
    )

    # Also preserve the original joblib file as a separate artefact
    mlflow.log_artifact(
        str(MODEL_PATH),
        artifact_path="model_files"
    )

    # Record available evaluation and explainability reports
    artifact_paths = [
        REPORTS_DIR / "optimised_training_results.json",
        REPORTS_DIR / "threshold_optimization_summary.json",
        REPORTS_DIR / "threshold_optimization_results.csv",
        REPORTS_DIR / "threshold_test_comparison.csv",
        REPORTS_DIR / "shap_feature_importance.csv",
        REPORTS_DIR / "shap_explanation_summary.json",
        FIGURES_DIR / "shap_summary.png",
        FIGURES_DIR / "shap_global_importance.png",
        FIGURES_DIR / "shap_local_explanation.png",
    ]

    print("\nLogging available reports and explainability artefacts...")

    for artifact_path in artifact_paths:
        if artifact_path.exists():
            mlflow.log_artifact(str(artifact_path))
            print(f"Logged: {artifact_path.name}")
        else:
            warnings.warn(
                f"Optional artefact not found and skipped: {artifact_path}"
            )

    # Display the run details
    run = mlflow.active_run()

    print("\nMLflow experiment run completed.")
    print("Experiment:", EXPERIMENT_NAME)
    print("Run ID:", run.info.run_id)
    print("Tracking URI:", mlflow.get_tracking_uri())

print("\nExperiment tracking finished successfully.")