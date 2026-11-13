from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient


# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "optimised_churn_model.joblib"
MLFLOW_DATABASE = PROJECT_ROOT / "mlflow.db"

EXPERIMENT_NAME = "ChurnOps-Customer-Churn"
REGISTERED_MODEL_NAME = "ChurnOps-Churn-Predictor"

mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DATABASE.as_posix()}"
)

mlflow.set_experiment(EXPERIMENT_NAME)

client = MlflowClient()


# --------------------------------------------------
# 2. Load the existing optimised model
# --------------------------------------------------

print("Loading the optimised churn prediction model...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")


# --------------------------------------------------
# 3. Log the model and register it
# --------------------------------------------------

print("\nLogging model for registration...")

with mlflow.start_run(run_name="ChurnOps-Model-Registration"):

    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        name="churn_prediction_model"
    )

    # Register the logged model
    registered_model = mlflow.register_model(
        model_uri=model_info.model_uri,
        name=REGISTERED_MODEL_NAME
    )

    print("\nModel registration successful.")
    print("Registered model:", registered_model.name)
    print("Model version:", registered_model.version)


# --------------------------------------------------
# 4. Add model documentation
# --------------------------------------------------

client.update_registered_model(
    name=REGISTERED_MODEL_NAME,
    description=(
        "Customer churn prediction model developed under "
        "the ChurnOps MLOps project. This model uses an "
        "optimised Logistic Regression pipeline trained "
        "on the IBM Telco Customer Churn dataset."
    )
)

client.update_model_version(
    name=REGISTERED_MODEL_NAME,
    version=registered_model.version,
    description=(
        "Initial registered version of the optimised "
        "Logistic Regression pipeline. Classification "
        "threshold: 0.54. Test F1-score: 0.6206. "
        "Test ROC-AUC: 0.8410."
    )
)

# Mark this version as the current candidate
client.set_registered_model_alias(
    name=REGISTERED_MODEL_NAME,
    alias="candidate",
    version=registered_model.version
)

print("\nModel documentation updated.")
print("Alias assigned: candidate")

print("\nREGISTRATION SUMMARY")
print("--------------------")
print("Model name:", REGISTERED_MODEL_NAME)
print("Version:", registered_model.version)
print("Alias: candidate")
print("Classification threshold: 0.54")
print("\nModel registration completed successfully.")