
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_predict
)


# =========================================================
# 1. PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "telco_churn_cleaned.csv"
)

MODEL_PATH = (
    PROJECT_ROOT / "models"
    / "optimised_churn_model.joblib"
)

REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. LOAD DATA AND OPTIMISED MODEL
# =========================================================

print("\nLoading cleaned dataset...")

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Churn", "customerID"])

y = df["Churn"].map({
    "No": 0,
    "Yes": 1
}).astype(int)

print("Dataset shape:", df.shape)

print("\nLoading optimised Logistic Regression model...")

optimised_model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")


# =========================================================
# 3. REPRODUCE ORIGINAL TRAIN-TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# =========================================================
# 4. GENERATE OUT-OF-FOLD TRAINING PROBABILITIES
# =========================================================

print("\nGenerating out-of-fold training probabilities...")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Clone the tuned model configuration.
# This creates an unfitted estimator with the same parameters.
model_for_cv = clone(optimised_model)

oof_probabilities = cross_val_predict(
    model_for_cv,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=-1
)[:, 1]

print("Out-of-fold probabilities generated.")


# =========================================================
# 5. FIND THE BEST CLASSIFICATION THRESHOLD
# =========================================================

print("\nEvaluating classification thresholds...")

thresholds = np.arange(0.20, 0.701, 0.01)

threshold_results = []

for threshold in thresholds:

    y_validation_pred = (
        oof_probabilities >= threshold
    ).astype(int)

    threshold_results.append({
        "threshold": float(threshold),
        "accuracy": accuracy_score(
            y_train,
            y_validation_pred
        ),
        "precision": precision_score(
            y_train,
            y_validation_pred,
            zero_division=0
        ),
        "recall": recall_score(
            y_train,
            y_validation_pred,
            zero_division=0
        ),
        "f1_score": f1_score(
            y_train,
            y_validation_pred,
            zero_division=0
        )
    })


threshold_df = pd.DataFrame(threshold_results)

# Select threshold with the highest out-of-fold F1-score.
best_index = threshold_df["f1_score"].idxmax()

best_threshold = float(
    threshold_df.loc[best_index, "threshold"]
)

print("\nBEST CLASSIFICATION THRESHOLD")
print("=" * 45)

print("Threshold:", round(best_threshold, 2))

print(
    "Validation F1-score:",
    round(
        threshold_df.loc[best_index, "f1_score"],
        4
    )
)


# =========================================================
# 6. EVALUATE ON THE UNTOUCHED TEST SET
# =========================================================

print("\nGenerating test-set probabilities...")

test_probabilities = optimised_model.predict_proba(
    X_test
)[:, 1]

# Default threshold.
baseline_predictions = (
    test_probabilities >= 0.50
).astype(int)

# Optimised threshold.
optimised_predictions = (
    test_probabilities >= best_threshold
).astype(int)


def evaluate_predictions(y_true, predictions, probabilities):

    return {
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),
        "f1_score": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),
        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),
        "pr_auc": float(
            average_precision_score(
                y_true,
                probabilities
            )
        )
    }


baseline_metrics = evaluate_predictions(
    y_test,
    baseline_predictions,
    test_probabilities
)

optimised_metrics = evaluate_predictions(
    y_test,
    optimised_predictions,
    test_probabilities
)


# =========================================================
# 7. DISPLAY PERFORMANCE COMPARISON
# =========================================================

comparison = pd.DataFrame(
    {
        "Default threshold (0.50)": baseline_metrics,
        f"Optimised threshold ({best_threshold:.2f})":
            optimised_metrics
    }
)

print("\nTEST SET PERFORMANCE COMPARISON")
print("=" * 60)

print(comparison.round(4))


print("\nDEFAULT THRESHOLD CONFUSION MATRIX")

baseline_cm = confusion_matrix(
    y_test,
    baseline_predictions
)

print(baseline_cm)


print("\nOPTIMISED THRESHOLD CONFUSION MATRIX")

optimised_cm = confusion_matrix(
    y_test,
    optimised_predictions
)

print(optimised_cm)


# =========================================================
# 8. PLOT F1-SCORE AGAINST THRESHOLD
# =========================================================

plt.figure(figsize=(9, 5))

plt.plot(
    threshold_df["threshold"],
    threshold_df["f1_score"],
    label="F1-score",
    linewidth=2
)

plt.axvline(
    best_threshold,
    linestyle="--",
    label=f"Selected threshold ({best_threshold:.2f})"
)

plt.axvline(
    0.50,
    linestyle=":",
    label="Default threshold (0.50)"
)

plt.title("Logistic Regression: F1-score vs Decision Threshold")
plt.xlabel("Classification Threshold")
plt.ylabel("F1-score")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "threshold_f1_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# =========================================================
# 9. PLOT PRECISION AND RECALL AGAINST THRESHOLD
# =========================================================

plt.figure(figsize=(9, 5))

plt.plot(
    threshold_df["threshold"],
    threshold_df["precision"],
    label="Precision",
    linewidth=2
)

plt.plot(
    threshold_df["threshold"],
    threshold_df["recall"],
    label="Recall",
    linewidth=2
)

plt.axvline(
    best_threshold,
    linestyle="--",
    label=f"Selected threshold ({best_threshold:.2f})"
)

plt.axvline(
    0.50,
    linestyle=":",
    label="Default threshold (0.50)"
)

plt.title("Precision-Recall Trade-off Across Thresholds")
plt.xlabel("Classification Threshold")
plt.ylabel("Score")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "threshold_precision_recall.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# =========================================================
# 10. SAVE THRESHOLD EXPERIMENT RESULTS
# =========================================================

threshold_df.to_csv(
    REPORT_DIR / "threshold_optimization_results.csv",
    index=False
)

comparison.to_csv(
    REPORT_DIR / "threshold_test_comparison.csv"
)


# =========================================================
# 11. SAVE TEST PREDICTIONS
# =========================================================

predictions_df = pd.DataFrame({
    "actual_churn": y_test.to_numpy(),
    "churn_probability": test_probabilities,
    "default_prediction": baseline_predictions,
    "optimised_prediction": optimised_predictions
})

predictions_df.to_csv(
    REPORT_DIR / "threshold_test_predictions.csv",
    index=False
)


# =========================================================
# 12. SAVE EXPERIMENT SUMMARY
# =========================================================

import json

results = {
    "model": "Optimised Logistic Regression",
    "threshold_selection_method": "5-fold out-of-fold predictions",
    "default_threshold": 0.50,
    "selected_threshold": best_threshold,
    "validation_f1": float(
        threshold_df.loc[best_index, "f1_score"]
    ),
    "default_test_metrics": baseline_metrics,
    "optimised_threshold_test_metrics": optimised_metrics,
    "default_confusion_matrix": baseline_cm.tolist(),
    "optimised_confusion_matrix": optimised_cm.tolist()
}

with open(
    REPORT_DIR / "threshold_optimization_summary.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(results, file, indent=4)


# =========================================================
# 13. COMPLETION SUMMARY
# =========================================================

print("\n" + "=" * 55)
print("CHURNOPS THRESHOLD OPTIMISATION COMPLETED")
print("=" * 55)

print("Default threshold: 0.50")
print("Selected threshold:", round(best_threshold, 2))

print("\nGenerated reports:")
print(REPORT_DIR / "threshold_optimization_results.csv")
print(REPORT_DIR / "threshold_test_comparison.csv")
print(REPORT_DIR / "threshold_test_predictions.csv")
print(REPORT_DIR / "threshold_optimization_summary.json")

print("\nGenerated plots:")
print(FIGURE_DIR / "threshold_f1_curve.png")
print(FIGURE_DIR / "threshold_precision_recall.png")
