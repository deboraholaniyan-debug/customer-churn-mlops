
from pathlib import Path
import json
import joblib
import mlflow
from mlflow.tracking import MlflowClient

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = PROJECT_ROOT / "mlflow.db"

DEPLOYMENT_DIR = PROJECT_ROOT / "deployment"
DEPLOYMENT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "ChurnOps-Churn-Predictor"
MODEL_ALIAS = "candidate"
CHURN_THRESHOLD = 0.54

# Connect to the existing local MLflow database
mlflow.set_tracking_uri(
    f"sqlite:///{DATABASE_PATH.as_posix()}"
)

client = MlflowClient()

print("\n=== RESOLVING REGISTERED MODEL ===")

# Resolve the current candidate alias
model_version = client.get_model_version_by_alias(
    MODEL_NAME,
    MODEL_ALIAS
)

print("Model:", model_version.name)
print("Version:", model_version.version)
print("Alias:", MODEL_ALIAS)

# Load the exact model referenced by the alias
model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

print("\nLoading model from MLflow...")
model = mlflow.sklearn.load_model(model_uri)

# Export model for Docker deployment
model_path = DEPLOYMENT_DIR / "model.joblib"
joblib.dump(model, model_path)

# Save deployment metadata
metadata = {
    "model_name": MODEL_NAME,
    "model_version": str(model_version.version),
    "model_alias": MODEL_ALIAS,
    "model_id": model_version.source.split("/")[-1],
    "source_run_id": model_version.run_id,
    "classification_threshold": CHURN_THRESHOLD
}

metadata_path = DEPLOYMENT_DIR / "model_metadata.json"

with open(metadata_path, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4)

print("\n=== DEPLOYMENT ARTEFACTS CREATED ===")
print("Model file:", model_path)
print("Metadata file:", metadata_path)
print("Model type:", type(model).__name__)
print("Threshold:", CHURN_THRESHOLD)

print("\nModel export completed successfully.")
