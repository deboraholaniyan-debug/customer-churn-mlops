
from pathlib import Path
import json
import joblib
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    average_precision_score
)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" /
    "telco_churn_cleaned.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "best_churn_model.joblib"
RESULTS_PATH = REPORT_DIR / "training_results.json"
PREDICTIONS_PATH = REPORT_DIR / "test_predictions.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\nLoading cleaned dataset...")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Cleaned dataset not found: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

# Preserve customer IDs for the prediction report
customer_ids = df["customerID"].copy()

# Separate features and target
X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

if y.isna().any():
    raise ValueError("Unexpected values found in Churn target.")

# Convert numeric columns to standard numerical types
numeric_features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

for column in numeric_features:
    X[column] = pd.to_numeric(
        X[column], errors="raise"
    ).astype("float64")

# Identify categorical features
categorical_features = [
    column for column in X.columns
    if column not in numeric_features
]

for column in categorical_features:
    X[column] = X[column].astype("object")

print("Numerical features:", len(numeric_features))
print("Categorical features:", len(categorical_features))

print("\nTarget distribution:")
print(y.value_counts())


# ============================================================
# 3. TRAIN-TEST SPLIT
# ============================================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test, ids_train, ids_test = (
    train_test_split(
        X,
        y,
        customer_ids,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# ============================================================
# 4. PREPROCESSING PIPELINE
# ============================================================

numeric_transformer = StandardScaler()

categorical_transformer = OneHotEncoder(
    handle_unknown="ignore"
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# ============================================================
# 5. DEFINE BASELINE MODELS
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=1
    )
}


# ============================================================
# 6. CROSS-VALIDATION
# ============================================================

print("\nStarting 5-fold stratified cross-validation...")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision"
}

cv_results = {}
pipelines = {}

for model_name, model in models.items():

    print(f"\nEvaluating {model_name}...")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model)
        ]
    )

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1
    )

    pipelines[model_name] = pipeline

    cv_results[model_name] = {}

    for metric in scoring:
        mean_score = scores[f"test_{metric}"].mean()
        std_score = scores[f"test_{metric}"].std()

        cv_results[model_name][metric] = {
            "mean": float(mean_score),
            "std": float(std_score)
        }

        print(
            f"{metric}: "
            f"{mean_score:.4f} (+/- {std_score:.4f})"
        )


# ============================================================
# 7. SELECT BEST MODEL USING MEAN CV F1
# ============================================================

best_model_name = max(
    cv_results,
    key=lambda name: cv_results[name]["f1"]["mean"]
)

best_pipeline = pipelines[best_model_name]

print("\nSelected model:", best_model_name)


# ============================================================
# 8. TRAIN SELECTED MODEL
# ============================================================

print("\nTraining selected model...")

best_pipeline.fit(X_train, y_train)


# ============================================================
# 9. EVALUATE ON TEST DATA
# ============================================================

print("\nEvaluating on the untouched test set...")

y_pred = best_pipeline.predict(X_test)

y_probability = best_pipeline.predict_proba(X_test)[:, 1]

test_metrics = {
    "accuracy": float(
        accuracy_score(y_test, y_pred)
    ),

    "precision": float(
        precision_score(y_test, y_pred, zero_division=0)
    ),

    "recall": float(
        recall_score(y_test, y_pred, zero_division=0)
    ),

    "f1_score": float(
        f1_score(y_test, y_pred, zero_division=0)
    ),

    "roc_auc": float(
        roc_auc_score(y_test, y_probability)
    ),

    "pr_auc": float(
        average_precision_score(y_test, y_probability)
    )
}

print("\nTEST SET PERFORMANCE")
print("=" * 45)

for metric, value in test_metrics.items():
    print(f"{metric}: {value:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["No Churn", "Churn"],
        zero_division=0
    )
)

conf_matrix = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(conf_matrix)


# ============================================================
# 10. MODEL EVALUATION VISUALISATIONS
# ============================================================

print("\nGenerating model evaluation plots...")

FIGURE_DIR = REPORT_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")

# ------------------------------------------------------------
# Plot 1: Confusion Matrix
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 5))

sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Churn", "Churn"],
    yticklabels=["No Churn", "Churn"],
    cbar=False,
    ax=ax
)

ax.set_title("Confusion Matrix - Logistic Regression", fontsize=14)
ax.set_xlabel("Predicted Class")
ax.set_ylabel("Actual Class")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ------------------------------------------------------------
# Plot 2: ROC Curve
# ------------------------------------------------------------

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(
    fpr,
    tpr,
    label=f"Logistic Regression (AUC = {test_metrics['roc_auc']:.3f})",
    linewidth=2
)

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

ax.set_title("Receiver Operating Characteristic (ROC) Curve")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(loc="lower right")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "roc_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ------------------------------------------------------------
# Plot 3: Precision-Recall Curve
# ------------------------------------------------------------

precision_values, recall_values, pr_thresholds = (
    precision_recall_curve(y_test, y_probability)
)

fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(
    recall_values,
    precision_values,
    label=f"Logistic Regression (AP = {test_metrics['pr_auc']:.3f})",
    linewidth=2
)

# Baseline equals the proportion of positive cases
baseline = y_test.mean()

ax.axhline(
    baseline,
    linestyle="--",
    label=f"Positive class baseline ({baseline:.3f})"
)

ax.set_title("Precision-Recall Curve")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.legend(loc="upper right")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "precision_recall_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ------------------------------------------------------------
# Plot 4: Cross-Validation Model Comparison
# ------------------------------------------------------------

model_names = list(cv_results.keys())

metrics_to_plot = ["f1", "roc_auc", "pr_auc"]

fig, ax = plt.subplots(figsize=(9, 6))

x = range(len(metrics_to_plot))
width = 0.35

for i, model_name in enumerate(model_names):

    means = [
        cv_results[model_name][metric]["mean"]
        for metric in metrics_to_plot
    ]

    stds = [
        cv_results[model_name][metric]["std"]
        for metric in metrics_to_plot
    ]

    positions = [
        value + (i - 0.5) * width
        for value in x
    ]

    ax.bar(
        positions,
        means,
        width,
        yerr=stds,
        capsize=5,
        label=model_name
    )

ax.set_title("Cross-Validation Model Performance Comparison")
ax.set_ylabel("Mean Score")
ax.set_xlabel("Evaluation Metric")
ax.set_xticks(list(x))
ax.set_xticklabels(["F1-score", "ROC-AUC", "PR-AUC"])
ax.set_ylim(0, 1)
ax.legend()

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


print("\nVisualisations saved successfully:")
print(FIGURE_DIR)


# ============================================================
# 11. SAVE MODEL
# ============================================================

joblib.dump(best_pipeline, MODEL_PATH)

print("\nTrained model saved to:")
print(MODEL_PATH)


# ============================================================
# 12. SAVE TEST PREDICTIONS
# ============================================================

predictions_df = pd.DataFrame({
    "customerID": ids_test.values,
    "actual_churn": y_test.values,
    "predicted_churn": y_pred,
    "churn_probability": y_probability
})

predictions_df["actual_churn"] = (
    predictions_df["actual_churn"].map({
        0: "No",
        1: "Yes"
    })
)

predictions_df["predicted_churn"] = (
    predictions_df["predicted_churn"].map({
        0: "No",
        1: "Yes"
    })
)

predictions_df.to_csv(
    PREDICTIONS_PATH,
    index=False
)


# ============================================================
# 13. SAVE EXPERIMENT RESULTS
# ============================================================

results = {
    "experiment": "ChurnOps Baseline Model Comparison",
    "dataset": "IBM Telco Customer Churn",
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
    "training_records": len(X_train),
    "testing_records": len(X_test),
    "selected_model": best_model_name,
    "cross_validation": cv_results,
    "test_metrics": test_metrics,
    "confusion_matrix": conf_matrix.tolist(),
    "classification_report": classification_report(
        y_test,
        y_pred,
        target_names=["No Churn", "Churn"],
        output_dict=True,
        zero_division=0
    )
}

with open(RESULTS_PATH, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4)


# ============================================================
# 14. TRAINING SUMMARY
# ============================================================

print("\n" + "=" * 55)
print("CHURNOPS MODEL TRAINING COMPLETED")
print("=" * 55)

print("Selected model:", best_model_name)
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("\nGenerated artefacts:")
print("Model:", MODEL_PATH)
print("Results:", RESULTS_PATH)
print("Predictions:", PREDICTIONS_PATH)
