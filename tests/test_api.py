
import os

import pytest
from fastapi.testclient import TestClient

# Ensure the API key exists before importing the application.
TEST_API_KEY = os.getenv("CHURNOPS_API_KEY")

if not TEST_API_KEY:
    raise RuntimeError(
        "CHURNOPS_API_KEY must be set before running API tests."
    )

from src.api import app


client = TestClient(app)

AUTH_HEADERS = {
    "X-API-Key": TEST_API_KEY
}

INVALID_AUTH_HEADERS = {
    "X-API-Key": "incorrect-test-api-key"
}


VALID_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "Yes",
    "tenure": 72,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No internet service",
    "OnlineBackup": "No internet service",
    "DeviceProtection": "No internet service",
    "TechSupport": "No internet service",
    "StreamingTV": "No internet service",
    "StreamingMovies": "No internet service",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 0,
    "TotalCharges": 0
}


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    result = response.json()

    assert result["status"] == "healthy"
    assert result["model_name"] == "ChurnOps-Churn-Predictor"
    assert result["model_alias"] == "candidate"


def test_prediction_requires_api_key():
    response = client.post(
        "/predict",
        json=VALID_CUSTOMER
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API key."


def test_prediction_rejects_invalid_api_key():
    response = client.post(
        "/predict",
        json=VALID_CUSTOMER,
        headers=INVALID_AUTH_HEADERS
    )

    assert response.status_code == 401


def test_prediction_endpoint():
    response = client.post(
        "/predict",
        json=VALID_CUSTOMER,
        headers=AUTH_HEADERS
    )

    assert response.status_code == 200

    result = response.json()

    assert result["predicted_churn"] in ["Yes", "No"]
    assert 0 <= result["churn_probability"] <= 1
    assert result["classification_threshold"] == 0.54
    assert result["model_name"] == "ChurnOps-Churn-Predictor"


def test_prediction_uses_configured_threshold():
    response = client.post(
        "/predict",
        json=VALID_CUSTOMER,
        headers=AUTH_HEADERS
    )

    assert response.status_code == 200

    result = response.json()

    expected_prediction = (
        "Yes"
        if result["churn_probability"] >= 0.54
        else "No"
    )

    assert result["predicted_churn"] == expected_prediction


def test_invalid_customer_category():
    invalid_customer = VALID_CUSTOMER.copy()
    invalid_customer["gender"] = "Unknown"

    response = client.post(
        "/predict",
        json=invalid_customer,
        headers=AUTH_HEADERS
    )

    assert response.status_code == 422


def test_invalid_tenure():
    invalid_customer = VALID_CUSTOMER.copy()
    invalid_customer["tenure"] = 100

    response = client.post(
        "/predict",
        json=invalid_customer,
        headers=AUTH_HEADERS
    )

    assert response.status_code == 422


def test_missing_required_field():
    invalid_customer = VALID_CUSTOMER.copy()
    del invalid_customer["MonthlyCharges"]

    response = client.post(
        "/predict",
        json=invalid_customer,
        headers=AUTH_HEADERS
    )

    assert response.status_code == 422
