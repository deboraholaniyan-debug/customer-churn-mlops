
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =========================================================
# 1. PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "telco_churn_cleaned.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. LOAD DATA
# =========================================================

print("\nLoading cleaned dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

TARGET = "Churn"
ID_COLUMN = "customerID"

# Remove customerID from predictive features.
X = df.drop(columns=[TARGET, ID_COLUMN])

# Convert target labels to binary values.
y = df[TARGET].map({
    "No": 0,
    "Yes": 1
})

if y.isna().any():
    raise ValueError("Unexpected values found in the Churn target.")

y = y.astype(int)

# Identify numerical and categorical columns.
numerical_features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

categorical_features = [
    col for col in X.columns
    if col not in numerical_features
]

print("Numerical features:", len(numerical_features))
print("Categorical features:", len(categorical_features))


# =========================================================
# 3. TRAIN-TEST SPLIT
# =========================================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# =========================================================
# 4. PREPROCESSING PIPELINE
# =========================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)


# =========================================================
# 5. BUILD LOGISTIC REGRESSION PIPELINE
# =========================================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=3000,
                random_state=42
            )
        )
    ]
)


# =========================================================
# 6. DEFINE HYPERPARAMETER SEARCH SPACE
# =========================================================

param_grid = {
    "classifier__C": [
        0.001,
        0.01,
        0.1,
        1,
        10,
        100
    ],

    "classifier__class_weight": [
        None,
        "balanced"
    ]
}
# Total combinations: 5 x 2 x 1 x 2 = 20.


# =========================================================
# 7. STRATIFIED CROSS-VALIDATION
# =========================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision"
}


# =========================================================
# 8. GRID SEARCH
# =========================================================

print("\nStarting Logistic Regression hyperparameter tuning...")
print("Testing 20 configurations with 5-fold cross-validation.")

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring=scoring,
    refit="f1",
    cv=cv,
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)

grid_search.fit(X_train, y_train)


# =========================================================
# 9. DISPLAY BEST CONFIGURATION
# =========================================================

best_model = grid_search.best_estimator_

print("\nBEST HYPERPARAMETERS")
print("=" * 50)

for parameter, value in grid_search.best_params_.items():
    print(f"{parameter}: {value}")

print(
    "\nBest mean cross-validation F1-score:",
    round(grid_search.best_score_, 4)
)


# =========================================================
# 10. EVALUATE ON UNTOUCHED TEST SET
# =========================================================

print("\nEvaluating optimised model on test data...")

y_pred = best_model.predict(X_test)
y_probability = best_model.predict_proba(X_test)[:, 1]

test_metrics = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision": float(precision_score(y_test, y_pred, zero_division=0)),
    "recall": float(recall_score(y_test, y_pred, zero_division=0)),
    "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, y_probability)),
    "pr_auc": float(average_precision_score(y_test, y_probability))
}

print("\nOPTIMISED MODEL TEST PERFORMANCE")
print("=" * 50)

for metric, value in test_metrics.items():
    print(f"{metric}: {value:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["No Churn", "Churn"]
    )
)

conf_matrix = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(conf_matrix)


# =========================================================
# 11. SAVE OPTIMISED MODEL
# =========================================================

model_path = MODEL_DIR / "optimised_churn_model.joblib"

joblib.dump(best_model, model_path)


# =========================================================
# 12. SAVE CROSS-VALIDATION RESULTS
# =========================================================

cv_results = pd.DataFrame(grid_search.cv_results_)

cv_results.to_csv(
    REPORT_DIR / "logistic_regression_tuning_results.csv",
    index=False
)


# =========================================================
# 13. SAVE TEST PREDICTIONS
# =========================================================

predictions = pd.DataFrame({
    "actual_churn": y_test.to_numpy(),
    "predicted_churn": y_pred,
    "churn_probability": y_probability
})

predictions.to_csv(
    REPORT_DIR / "optimised_test_predictions.csv",
    index=False
)


# =========================================================
# 14. SAVE EVALUATION SUMMARY
# =========================================================

results = {
    "model": "Logistic Regression",
    "experiment": "Hyperparameter optimisation",
    "best_parameters": grid_search.best_params_,
    "best_cv_f1": float(grid_search.best_score_),
    "test_metrics": test_metrics,
    "confusion_matrix": conf_matrix.tolist(),
    "classification_report": classification_report(
        y_test,
        y_pred,
        target_names=["No Churn", "Churn"],
        output_dict=True
    )
}

with open(
    REPORT_DIR / "optimised_training_results.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(results, file, indent=4)


# =========================================================
# 15. COMPLETION SUMMARY
# =========================================================

print("\n" + "=" * 55)
print("CHURNOPS HYPERPARAMETER OPTIMISATION COMPLETED")
print("=" * 55)

print("Best model: Logistic Regression")

print("Optimised model saved to:")
print(model_path)

print("\nGenerated reports:")
print(REPORT_DIR / "logistic_regression_tuning_results.csv")
print(REPORT_DIR / "optimised_test_predictions.csv")
print(REPORT_DIR / "optimised_training_results.json")
