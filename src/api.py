
from pathlib import Path
from typing import Literal

import mlflow
import mlflow.sklearn
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# --------------------------------------------------
# 1. Project and MLflow configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MLFLOW_DATABASE = PROJECT_ROOT / "mlflow.db"

mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DATABASE.as_posix()}"
)

MODEL_NAME = "ChurnOps-Churn-Predictor"
MODEL_ALIAS = "candidate"
MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

CHURN_THRESHOLD = 0.54


# --------------------------------------------------
# 2. Load the registered model
# --------------------------------------------------

model = mlflow.sklearn.load_model(MODEL_URI)

# Retrieve the registered model version for API metadata.
client = mlflow.tracking.MlflowClient()

try:
    registered_version = client.get_model_version_by_alias(
        MODEL_NAME, MODEL_ALIAS
    )
    MODEL_VERSION = registered_version.version
except Exception:
    MODEL_VERSION = "unknown"


# --------------------------------------------------
# 3. Define the customer input schema
# --------------------------------------------------

class CustomerFeatures(BaseModel):
    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=72)

    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal[
        "No phone service", "No", "Yes"
    ]

    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal[
        "No internet service", "No", "Yes"
    ]
    OnlineBackup: Literal[
        "No internet service", "No", "Yes"
    ]
    DeviceProtection: Literal[
        "No internet service", "No", "Yes"
    ]
    TechSupport: Literal[
        "No internet service", "No", "Yes"
    ]
    StreamingTV: Literal[
        "No internet service", "No", "Yes"
    ]
    StreamingMovies: Literal[
        "No internet service", "No", "Yes"
    ]

    Contract: Literal[
        "Month-to-month", "One year", "Two year"
    ]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]

    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


# --------------------------------------------------
# 4. Initialise FastAPI
# --------------------------------------------------

app = FastAPI(
    title="ChurnOps Customer Churn Prediction API",
    description=(
        "An MLflow-backed API for predicting customer churn "
        "using the registered ChurnOps machine-learning model."
    ),
    version="1.0.0"
)


# --------------------------------------------------
# 5. Health-check endpoint
# --------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "ChurnOps API",
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_alias": MODEL_ALIAS
    }


# --------------------------------------------------
# 6. Customer churn prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict_churn(customer: CustomerFeatures):

    try:
        # Convert validated input into a DataFrame.
        input_data = pd.DataFrame(
            [customer.model_dump()]
        )

        # Obtain the probability of class 1 (Churn = Yes).
        churn_probability = float(
            model.predict_proba(input_data)[0, 1]
        )

        # Apply the optimised classification threshold.
        predicted_churn = (
            churn_probability >= CHURN_THRESHOLD
        )

        return {
            "predicted_churn": (
                "Yes" if predicted_churn else "No"
            ),
            "churn_probability": round(
                churn_probability, 4
            ),
            "classification_threshold": CHURN_THRESHOLD,
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )
