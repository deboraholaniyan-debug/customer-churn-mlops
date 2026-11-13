
# ChurnOps

## End-to-End MLOps Pipeline for Customer Churn Prediction

ChurnOps is an end-to-end machine learning project designed to predict customer churn, explain model predictions, and establish a foundation for model monitoring and automated retraining.

## Project Objectives

- Develop a machine learning model for customer churn prediction.
- Evaluate classification performance using multiple metrics.
- Optimise hyperparameters and classification thresholds.
- Explain model predictions using SHAP.
- Implement experiment tracking and model management.
- Deploy the model through a REST API.
- Containerise the application using Docker.
- Implement model monitoring and automated retraining workflows.

## Dataset

IBM Telco Customer Churn dataset.

Source: https://www.kaggle.com/datasets/blastchar/ibm-telco-customer-churn

The dataset contains 7,043 customer records and 21 original variables.

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- SHAP
- Joblib

Planned additions:
- MLflow
- FastAPI
- Docker
- Pytest
- GitHub Actions
- Evidently
- DVC

## Current Implementation

1. Data cleaning and validation.
2. Exploratory data analysis.
3. Logistic Regression and Random Forest training.
4. Hyperparameter optimisation.
5. Classification threshold optimisation.
6. SHAP model explainability.

## Project Structure

```text
customer-churn-mlops/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── reports/
│   └── figures/
├── src/
│   ├── data_cleaning.py
│   ├── train.py
│   ├── tune_logistic_regression.py
│   ├── threshold_optimization.py
│   └── explain_model.py
├── eda.py
├── README.md
└── .gitignore
```

## Model Evaluation

The current threshold-optimised Logistic Regression model achieved:

| Metric | Result |
|---|---:|
| Accuracy | 75.44% |
| Precision | 52.60% |
| Recall | 75.67% |
| F1-score | 62.06% |
| ROC-AUC | 84.10% |
| PR-AUC | 63.33% |

The classification threshold was selected using out-of-fold predictions on the training partition.

## Project Status

In development. MLflow experiment tracking, API deployment, containerisation, automated testing and model monitoring are planned.
