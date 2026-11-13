
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = PROJECT_ROOT / "mlflow.db"

mlflow.set_tracking_uri(
    f"sqlite:///{DATABASE_PATH.as_posix()}"
)

client = MlflowClient()

model_name = "ChurnOps-Churn-Predictor"
alias = "candidate"

# Retrieve the registered candidate version
version = client.get_model_version_by_alias(
    model_name,
    alias
)

print("\n=== REGISTERED MODEL VERSION ===")
print("Name:", version.name)
print("Version:", version.version)
print("Source:", version.source)
print("Run ID:", version.run_id)

# Inspect the MLflow 3 logged-model record
model_id = version.source.split("/")[-1]

print("\n=== LOGGED MODEL DETAILS ===")

logged_model = client.get_logged_model(model_id)

print("Model ID:", logged_model.model_id)
print("Artifact location:", logged_model.artifact_location)
print("Source run ID:", logged_model.source_run_id)
print("Status:", logged_model.status)

# Confirm that the candidate can still be loaded
print("\n=== MODEL LOADING TEST ===")

model = mlflow.sklearn.load_model(
    f"models:/{model_name}@{alias}"
)

print("Model loaded successfully:", type(model).__name__)
print("\nInspection completed.")
