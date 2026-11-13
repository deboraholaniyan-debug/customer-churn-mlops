
from pathlib import Path

import mlflow
import mlflow.sklearn


# Resolve the project root and MLflow database.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MLFLOW_DATABASE = PROJECT_ROOT / "mlflow.db"

# Connect to the same MLflow tracking database.
mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DATABASE.as_posix()}"
)

MODEL_NAME = "ChurnOps-Churn-Predictor"
MODEL_ALIAS = "candidate"

MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

print(f"Loading registered model: {MODEL_URI}")

try:
    model = mlflow.sklearn.load_model(MODEL_URI)

    print("\nModel loaded successfully.")
    print(f"Model type: {type(model).__name__}")

    if hasattr(model, "predict_proba"):
        print("Probability prediction: available")
    else:
        print("Probability prediction: unavailable")

except Exception as error:
    print("\nModel loading failed.")
    print(f"Error: {error}")
    raise
