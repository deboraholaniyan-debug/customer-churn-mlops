# ChurnOps

## An End-to-End MLOps Pipeline for Customer Churn Prediction, Explainability and Deployment

ChurnOps is an end-to-end Machine Learning Operations (MLOps) project that demonstrates the development, evaluation, explainability, management and deployment of a customer churn prediction model.

Built using the IBM Telco Customer Churn dataset, the project implements a reproducible machine learning workflow covering data preprocessing, exploratory data analysis, model development, hyperparameter optimisation, classification threshold selection, SHAP-based explainability, MLflow experiment tracking and model registration.

The trained model is served through a FastAPI REST API, packaged using Docker and published to the GitHub Container Registry (GHCR). Automated testing and container image publishing are integrated into GitHub Actions.

The project also establishes a foundation for subsequent implementation of production-oriented model monitoring, data versioning and automated retraining.

## Project Objectives

- Develop and evaluate machine learning models for customer churn prediction.
- Implement reproducible data preprocessing and exploratory data analysis.
- Compare classification algorithms using appropriate evaluation metrics.
- Optimise model hyperparameters and classification thresholds.
- Apply SHAP to interpret model predictions and identify influential features.
- Implement experiment tracking and model version management using MLflow.
- Expose model inference through a RESTful API using FastAPI.
- Containerise the prediction service using Docker.
- Implement automated testing and continuous integration using GitHub Actions.
- Publish versioned Docker images to GitHub Container Registry.
- Establish a foundation for model monitoring, data versioning and automated retraining.

## Dataset

**IBM Telco Customer Churn Dataset**

Source: [Kaggle – IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/ibm-telco-customer-churn)

The dataset contains 7,043 customer records and 21 original variables describing customer demographics, account information, subscribed services, billing arrangements and churn status.

The target variable is `Churn`, which indicates whether a customer discontinued the service.

### Data preprocessing

The preprocessing workflow includes:

- Inspection of missing values and duplicate records.
- Removal of the customer identifier from predictive features.
- Conversion of `TotalCharges` into a numerical variable.
- Treatment of 11 missing `TotalCharges` values associated with zero-tenure customers.
- Validation of the cleaned dataset.

The cleaned dataset contains 7,043 records with no remaining missing values.

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Data manipulation | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Machine learning | Scikit-learn |
| Model explainability | SHAP |
| Model serialisation | Joblib |
| Experiment tracking | MLflow |
| Model registry | MLflow Model Registry |
| API development | FastAPI, Pydantic |
| API server | Uvicorn |
| Testing | Pytest |
| Containerisation | Docker, Docker Compose |
| CI/CD | GitHub Actions |
| Container registry | GitHub Container Registry (GHCR) |
| Version control | Git and GitHub |

## Implemented Pipeline

### 1. Data cleaning and preprocessing

The raw dataset is inspected, cleaned and transformed into a format suitable for machine learning. Numerical and categorical variables are handled through a Scikit-learn preprocessing pipeline.

### 2. Exploratory data analysis

Exploratory analysis examines customer churn distribution, tenure, monthly charges, total charges, contract types, payment methods and other relevant customer characteristics.

Visualisations are generated and stored in the project reports directory.

### 3. Model development

Logistic Regression and Random Forest classifiers are evaluated using a consistent training and validation strategy.

The Logistic Regression model is selected based on cross-validation performance.

### 4. Hyperparameter optimisation

GridSearchCV is used to optimise Logistic Regression hyperparameters, including regularisation strength and class weighting.

### 5. Classification threshold optimisation

The classification threshold is selected using out-of-fold predictions from the training partition rather than optimising directly against the test set.

The selected threshold is 0.54.

### 6. Model explainability

SHAP is applied to the optimised Logistic Regression model to examine feature contributions to churn predictions.

The analysis identifies customer tenure, contract type, total charges and internet service characteristics among the influential predictive features.

SHAP contributions indicate model attribution, not causal relationships.

### 7. Experiment tracking and model management

MLflow is used to record model parameters, evaluation metrics and relevant experiment artifacts.

The selected model is registered as `ChurnOps-Churn-Predictor`, with version 2 assigned the `candidate` alias.

### 8. API development

A FastAPI service exposes the trained model through REST endpoints.

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Checks service health and model metadata |
| `/predict` | POST | Generates a customer churn prediction |
| `/docs` | GET | Provides interactive API documentation |

The prediction response includes the predicted churn class, estimated churn probability, classification threshold and model version.

### 9. Containerisation and deployment

The inference service is packaged using Docker. Docker Compose provides a convenient local deployment configuration, including port mapping, restart policy and container health checks.

The Docker image is published to GitHub Container Registry.

**Published image:**

`ghcr.io/deboraholaniyan-debug/customer-churn-mlops:latest`

Version-specific image tags are also published to support traceability.

### 10. Automated testing and CI/CD

GitHub Actions automates the testing and Docker build workflow.

The pipeline installs project dependencies, executes API tests and builds the Docker image. On pushes to the main branch, the workflow also publishes the image to GHCR.

## Model Evaluation

The threshold-optimised Logistic Regression model achieved the following results on the held-out test set:

| Metric | Result |
|---|---:|
| Accuracy | 75.44% |
| Precision | 52.60% |
| Recall | 75.67% |
| F1-score | 62.06% |
| ROC-AUC | 84.10% |
| PR-AUC | 63.33% |

The classification threshold of 0.54 was selected using out-of-fold predictions from the training partition.

Compared with the default threshold of 0.50, the selected threshold produced a modest improvement in F1-score, from 61.76% to 62.06%, with a corresponding reduction in recall.

ROC-AUC and PR-AUC remained unchanged because threshold selection changes the classification decision boundary rather than the model's underlying probability ranking.

## Live Inference Example

The deployed API was tested using the published GHCR image.

Example response:

```json
{
  "predicted_churn": "Yes",
  "churn_probability": 0.8294,
  "classification_threshold": 0.54,
  "model_name": "ChurnOps-Churn-Predictor",
  "model_version": "2"
}
```

This demonstrates a successful inference request against the containerised service. The probability represents the model's estimate for the supplied customer record, not a guaranteed future outcome.

## Project Structure

```text
customer-churn-mlops/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── data_cleaning.py
│   ├── train.py
│   ├── tune_logistic_regression.py
│   ├── threshold_optimization.py
│   ├── explain_model.py
│   ├── track_experiments.py
│   ├── register_model.py
│   ├── export_model.py
│   └── api.py
│
├── models/
│
├── deployment/
│   ├── model.joblib
│   └── model_metadata.json
│
├── reports/
│   └── figures/
│
├── tests/
│   └── test_api.py
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── Dockerfile
├── compose.yaml
├── .dockerignore
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
├── eda.py
├── README.md
└── .gitignore
```

*Note: This structure represents the principal project components; additional files and directories support experiment tracking and local development.*

## Running the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/deboraholaniyan-debug/customer-churn-mlops.git

cd customer-churn-mlops
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Linux / WSL:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements-dev.txt
```

### 4. Run automated tests

```bash
pytest tests/ -v
```

### 5. Start the API using Docker Compose

```bash
docker compose up --build -d
```

The API will be accessible at:

- API health: `http://localhost:8000/health`
- Interactive documentation: `http://localhost:8000/docs`

### 6. Stop the service

```bash
docker compose down
```

## Current Project Status

**Implemented and tested**

- Data cleaning and preprocessing.
- Exploratory data analysis.
- Model development and evaluation.
- Hyperparameter optimisation.
- Classification threshold optimisation.
- SHAP explainability.
- MLflow experiment tracking and model registration.
- Model export and version metadata.
- FastAPI inference service.
- Docker and Docker Compose deployment.
- Automated API tests.
- GitHub Actions CI workflow.
- GHCR image publishing.
- Container health and restart validation.

**Planned enhancements**

- Data versioning using DVC.
- Data and model drift monitoring using Evidently.
- Automated model retraining and evaluation workflows.
- Model performance monitoring over time.
- Deployment to a public cloud environment.
- Enhanced operational dashboards and observability.

## Project Repository

[GitHub – ChurnOps](https://github.com/deboraholaniyan-debug/customer-churn-mlops)

## Research and Engineering Direction

ChurnOps is being developed as a practical demonstration of how machine learning models can move beyond experimentation into reproducible, explainable and deployable services.

The broader engineering objective is to establish a maintainable lifecycle in which model development, experiment tracking, deployment, monitoring and model updates are integrated into a consistent workflow.
