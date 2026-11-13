
from pathlib import Path
import json

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal


# --------------------------------------------------
# 1. Project and deployment configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEPLOYMENT_DIR = PROJECT_ROOT / "deployment"

MODEL_PATH = DEPLOYMENT_DIR / "model.joblib"
METADATA_PATH = DEPLOYMENT_DIR / "model_metadata.json"


# --------------------------------------------------
# 2. Load deployment metadata
# --------------------------------------------------

if not METADATA_PATH.exists():
    raise FileNotFoundError(
        f"Model metadata not found: {METADATA_PATH}"
    )

with open(METADATA_PATH, "r", encoding="utf-8") as file:
    MODEL_METADATA = json.load(file)


MODEL_NAME = MODEL_METADATA["model_name"]
MODEL_VERSION = MODEL_METADATA["model_version"]
MODEL_ALIAS = MODEL_METADATA["model_alias"]

CHURN_THRESHOLD = float(
    MODEL_METADATA["classification_threshold"]
)


# --------------------------------------------------
# 3. Load the exported machine-learning model
# --------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# 4. Define the customer input schema
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
# 5. Initialise FastAPI
# --------------------------------------------------

app = FastAPI(
    title="ChurnOps Customer Churn Prediction API",
    description=(
        "A container-ready API for predicting customer "
        "churn using a versioned machine-learning model "
        "exported from the MLflow Model Registry."
    ),
    version="1.0.0"
)


# --------------------------------------------------
# 6. Health-check endpoint
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
# 7. Customer churn prediction endpoint
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

        # Apply the threshold recorded in deployment metadata.
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
