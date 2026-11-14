# ChurnOps

## An End-to-End MLOps Platform for Customer Churn Prediction, Explainability and Monitoring

ChurnOps is an end-to-end Machine Learning Operations (MLOps) project designed to demonstrate how machine learning models can be developed, evaluated, interpreted, deployed and monitored within an integrated application.

The platform uses the IBM Telco Customer Churn dataset to develop a predictive model that estimates the likelihood of customer attrition. It combines machine learning experimentation, model explainability, experiment tracking, API development, containerisation, automated testing and operational monitoring.

ChurnOps also includes a customer-facing web application built with Next.js. Users can register, log in and submit individual customer information through a structured prediction form. The application communicates with a deployed FastAPI prediction service through a server-side proxy, keeping the prediction API credentials away from the browser.

The project demonstrates the integration of machine learning engineering, backend development, frontend application development and MLOps practices in a unified workflow.

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
- Develop a customer-facing prediction application using Next.js.
- Implement user authentication using Supabase.
- Integrate the web application with the deployed prediction API.
- Establish data drift monitoring and operational reporting components.

## System Architecture

ChurnOps consists of four main components:

1. **Machine learning and MLOps layer:** Handles data preparation, model development, evaluation, explainability, experiment tracking and model registration.
2. **Prediction API:** Serves the trained model through FastAPI and is deployed on Render.
3. **Web application:** Provides the customer-facing interface through which authenticated users submit customer information and receive prediction results.
4. **Monitoring layer:** Contains reference datasets, simulated production data, drift assessment scripts and monitoring reports.

### Application workflow

```text
                 CUSTOMER
                     |
                     v
          Next.js Web Application
             (Frontend)
                     |
                     v
            Supabase Authentication
                     |
                     v
          Next.js Server-side Proxy
                     |
                     v
             FastAPI Prediction API
                  (Render)
                     |
                     v
             Churn Prediction Model
                     |
                     v
          Prediction Result Returned
                     |
                     v
             Customer Dashboard


       Monitoring and MLOps Workflow

       Reference Dataset
              |
              v
       Simulated Production Data
              |
              v
       Drift Assessment Scripts
              |
              v
       Monitoring Reports
```

The prediction API key is stored in the server-side environment configuration and is not exposed to the browser.

## Dataset

**IBM Telco Customer Churn Dataset**

Source: [Kaggle – IBM Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/ibm-telco-customer-churn)

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
| Container registry | GitHub Container Registry |
| Frontend | Next.js, React, TypeScript |
| Styling | Tailwind CSS |
| Authentication and database | Supabase |
| Backend deployment | Render |
| Version control | Git and GitHub |

## Machine Learning Pipeline

### 1. Data cleaning and preprocessing

The raw dataset is inspected, cleaned and transformed into a format suitable for machine learning. Numerical and categorical variables are handled through a Scikit-learn preprocessing pipeline.

### 2. Exploratory data analysis

Exploratory analysis examines customer churn distribution, tenure, monthly charges, total charges, contract types, payment methods and other relevant customer characteristics.

Visualisations are generated and stored in the project reports directory.

### 3. Model development

Logistic Regression and Random Forest classifiers are evaluated using a consistent training and validation strategy.

The Logistic Regression model is selected for the deployed prediction service.

### 4. Hyperparameter optimisation

GridSearchCV is used to optimise Logistic Regression hyperparameters, including regularisation strength and class weighting.

### 5. Classification threshold optimisation

The classification threshold is selected using out-of-fold predictions from the training partition rather than optimising directly against the test set.

The selected classification threshold is **0.54**.

### 6. Model explainability

SHAP is applied to the optimised Logistic Regression model to examine feature contributions to churn predictions.

The analysis identifies customer tenure, contract type, total charges and internet service characteristics among the influential predictive features.

SHAP contributions represent model attribution and should not be interpreted as evidence of causal relationships.

### 7. Experiment tracking and model management

MLflow is used to record model parameters, evaluation metrics and relevant experiment artefacts.

The selected model is registered as:

`ChurnOps-Churn-Predictor`

The deployed model is version 2.

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

ROC-AUC and PR-AUC evaluate the model's ranking performance across classification thresholds, while precision, recall, F1-score and accuracy depend on the selected decision threshold.

## Backend API

The prediction service is developed using FastAPI and deployed on Render.

**Deployed API:**

https://customer-churn-mlops-sha-38eb8e5.onrender.com

### API endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Checks service health and model metadata |
| `/predict` | POST | Generates a customer churn prediction |
| `/docs` | GET | Provides interactive API documentation |

The `/predict` endpoint requires API-key authentication through the `X-API-Key` request header.

The prediction response includes:

- Predicted churn class.
- Estimated churn probability.
- Classification threshold.
- Model name.
- Model version.

### Example prediction response

```json
{
  "predicted_churn": "Yes",
  "churn_probability": 0.8294,
  "classification_threshold": 0.54,
  "model_name": "ChurnOps-Churn-Predictor",
  "model_version": "2"
}
```

The example illustrates the response format. The probability represents the model's estimate for the supplied customer record, not a guaranteed future outcome.

## Customer-facing Web Application

ChurnOps includes a web application developed with Next.js, React, TypeScript and Tailwind CSS.

The application provides an interface for authenticated users to submit individual customer information and obtain churn predictions from the deployed machine learning model.

### Implemented functionality

- User registration.
- User login and session management.
- Protected dashboard and prediction routes.
- Structured customer information form.
- Individual customer churn prediction.
- Customer-friendly prediction result presentation.
- Churn probability visualisation.
- Classification outcome and threshold display.
- Server-side communication with the FastAPI service.

### Authentication and security

Supabase provides authentication and session management.

The frontend uses a server-side API proxy to communicate with the FastAPI service. The prediction API key is stored in server-side environment variables rather than exposed in client-side code.

The application uses environment variables for service configuration. Credentials and secrets must not be committed to the Git repository.

**Deployment status:** The frontend implementation is available in the repository. Public deployment through Vercel remains a separate deployment step.

## Model Monitoring and Drift Assessment

The project includes monitoring components designed to support the assessment of changes between reference data and simulated production data.

The monitoring implementation includes:

- Reference monitoring datasets.
- Simulated production batch data.
- Data preparation scripts.
- Data drift assessment scripts.
- Monitoring run assessment.
- Generated HTML and JSON monitoring reports.
- Prediction drift comparison outputs.

Monitoring-related files are organised under:

- `data/monitoring/`
- `data/simulated/`
- `reports/monitoring/`
- `src/`

These components establish a foundation for operational monitoring. They should not be interpreted as evidence of continuous live monitoring or automated model retraining.

## Containerisation and CI/CD

The FastAPI inference service is packaged using Docker.

Docker Compose provides a convenient local deployment configuration, including port mapping, restart policy and container health checks.

The Docker image is published to GitHub Container Registry (GHCR).

### Published container image

`ghcr.io/deboraholaniyan-debug/customer-churn-mlops:latest`

Version-specific image tags are also published to support traceability.

### GitHub Actions

GitHub Actions automates the backend testing and Docker build workflow.

The pipeline includes:

- Dependency installation.
- Automated API tests using Pytest.
- Docker image build.
- Container health and integration checks.
- Container image publishing to GHCR.
- Render deployment triggering.

## Project Structure

```text
customer-churn-mlops/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── monitoring/
│   │   ├── reference_data.csv
│   │   └── reference_labels.csv
│   └── simulated/
│       └── simulated_production_batch_001.csv
│
├── frontend/
│   ├── app/
│   │   ├── (protected)/
│   │   │   ├── layout.tsx
│   │   │   ├── page.tsx
│   │   │   └── new-prediction/
│   │   │       └── page.tsx
│   │   ├── api/
│   │   │   └── predict/
│   │   │       └── route.ts
│   │   ├── auth/
│   │   │   └── callback/
│   │   │       └── route.ts
│   │   ├── login/
│   │   ├── signup/
│   │   ├── globals.css
│   │   └── layout.tsx
│   ├── lib/
│   │   └── supabase/
│   │       ├── client.ts
│   │       ├── proxy.ts
│   │       └── server.ts
│   ├── package.json
│   ├── package-lock.json
│   └── proxy.ts
│
├── src/
│   ├── assess_monitoring_run.py
│   ├── monitor_data_drift.py
│   └── prepare_monitoring_data.py
│
├── deployment/
│   ├── model.joblib
│   └── model_metadata.json
│
├── reports/
│   ├── figures/
│   └── monitoring/
│       ├── data_drift_report.html
│       ├── data_drift_report.json
│       ├── monitoring_alert.json
│       ├── monitoring_summary.json
│       └── prediction_drift_comparison.csv
│
├── tests/
├── .github/
│   └── workflows/
├── Dockerfile
├── compose.yaml
├── requirements.txt
├── requirements-dev.txt
├── README.md
└── .gitignore
```

*The structure above highlights the principal components. Additional files support model development, API deployment, testing and local development.*

## Running the Backend Locally

### 1. Clone the repository

```bash
git clone https://github.com/deboraholaniyan-debug/customer-churn-mlops.git

cd customer-churn-mlops
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Linux or WSL:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

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

The prediction endpoint requires a valid API key configured for the local service.

### 6. Stop the service

```bash
docker compose down
```

## Frontend Configuration

The frontend is located in the `frontend/` directory.

### 1. Navigate to the frontend

```bash
cd frontend
```

### 2. Install dependencies

```bash
npm install
```

### 3. Configure environment variables

Create a `.env.local` file in the `frontend/` directory.

Configure the following variables using your own service credentials:

```env
NEXT_PUBLIC_SUPABASE_URL=your_supabase_project_url
NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY=your_supabase_publishable_key
CHURN_API_URL=your_render_api_url
CHURN_API_KEY=your_prediction_api_key
```

Do not commit `.env.local` or expose the prediction API key through a `NEXT_PUBLIC_` variable.

### 4. Start the development server

```bash
npm run dev
```

Open:

`http://localhost:3000`

Supabase authentication must also be configured with the appropriate application URL and authentication redirect settings.

## Current Project Status

### Implemented components

- Data cleaning and preprocessing.
- Exploratory data analysis.
- Model development and evaluation.
- Hyperparameter optimisation.
- Classification threshold optimisation.
- SHAP-based model explainability.
- MLflow experiment tracking and model registration.
- Model export and version metadata.
- FastAPI prediction service.
- API-key authentication.
- Docker and Docker Compose configuration.
- Automated API testing.
- GitHub Actions CI/CD workflow.
- GHCR image publishing.
- Render backend deployment.
- Next.js customer-facing frontend.
- Supabase user registration and authentication.
- Individual customer prediction interface.
- Server-side prediction API integration.
- Monitoring reference datasets.
- Simulated production data.
- Data drift assessment scripts and monitoring reports.

### Further development opportunities

- Deploy the frontend to Vercel.
- Implement tenant-aware organisation management.
- Introduce persistent customer prediction history.
- Integrate monitoring results into the web dashboard.
- Implement scheduled monitoring and alerting.
- Introduce automated model retraining and evaluation workflows.
- Add model performance monitoring using verified production outcomes.
- Implement data versioning using DVC.
- Extend the API to support batch predictions.

## Project Repository

**GitHub:**  
https://github.com/deboraholaniyan-debug/customer-churn-mlops

**Backend API:**  
https://customer-churn-mlops-sha-38eb8e5.onrender.com

## Research and Engineering Direction

ChurnOps demonstrates how machine learning models can move beyond experimentation into reproducible, explainable and deployable services.

The project brings together predictive analytics, model governance, API engineering, application development and monitoring within a practical MLOps architecture.

Its broader engineering direction is to establish a maintainable machine learning lifecycle in which experimentation, model management, deployment, monitoring and future model updates operate as connected components.
