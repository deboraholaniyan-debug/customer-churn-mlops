
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


# ==================================================
# 1. PROJECT PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "telco_churn_cleaned.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "optimised_churn_model.joblib"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. LOAD DATA AND TRAINED MODEL
# ==================================================

print("Loading cleaned dataset and trained model...")

df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

X = df.drop(columns=["Churn", "customerID"])
y = df["Churn"].map({"No": 0, "Yes": 1})

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Dataset loaded successfully.")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# ==================================================
# 3. TRANSFORM FEATURES
# ==================================================

print("\nTransforming features...")

preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

X_train_transformed = preprocessor.transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

if hasattr(X_train_transformed, "toarray"):
    X_train_transformed = X_train_transformed.toarray()

if hasattr(X_test_transformed, "toarray"):
    X_test_transformed = X_test_transformed.toarray()

X_train_transformed = np.asarray(X_train_transformed)
X_test_transformed = np.asarray(X_test_transformed)

feature_names = preprocessor.get_feature_names_out()

print("Training matrix:", X_train_transformed.shape)
print("Testing matrix:", X_test_transformed.shape)
print("Number of transformed features:", len(feature_names))


# ==================================================
# 4. INITIALISE SHAP EXPLAINER
# ==================================================

print("\nInitialising SHAP LinearExplainer...")

rng = np.random.default_rng(42)

background_size = min(100, len(X_train_transformed))

background_indices = rng.choice(
    len(X_train_transformed),
    size=background_size,
    replace=False
)

background = X_train_transformed[background_indices]

explanation_size = min(200, len(X_test_transformed))
X_explain = X_test_transformed[:explanation_size]

explainer = shap.LinearExplainer(
    classifier,
    background
)

shap_explanation = explainer(X_explain)

print("SHAP values calculated successfully.")


# ==================================================
# 5. GLOBAL FEATURE IMPORTANCE
# ==================================================

print("\nCalculating global feature importance...")

mean_abs_shap = np.abs(shap_explanation.values).mean(axis=0)

importance_df = pd.DataFrame({
    "feature": feature_names,
    "mean_absolute_shap": mean_abs_shap
}).sort_values(
    by="mean_absolute_shap",
    ascending=False
)

importance_df.to_csv(
    REPORTS_DIR / "shap_feature_importance.csv",
    index=False
)

print("\nTOP 15 INFLUENTIAL FEATURES")
print(importance_df.head(15).to_string(index=False))


# ==================================================
# 6. SHAP SUMMARY PLOT
# ==================================================

print("\nGenerating SHAP summary plot...")

shap.summary_plot(
    shap_explanation.values,
    X_explain,
    feature_names=feature_names,
    max_display=20,
    show=False
)

plt.title("SHAP Summary - Customer Churn Prediction")
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==================================================
# 7. GLOBAL IMPORTANCE BAR PLOT
# ==================================================

print("Generating global importance plot...")

shap.summary_plot(
    shap_explanation.values,
    X_explain,
    feature_names=feature_names,
    plot_type="bar",
    max_display=20,
    show=False
)

plt.title("Global SHAP Feature Importance")
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "shap_global_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==================================================
# 8. LOCAL CUSTOMER EXPLANATION
# ==================================================

print("Generating individual customer explanation...")

shap.plots.waterfall(
    shap_explanation[0],
    max_display=15,
    show=False
)

plt.title("SHAP Explanation for an Individual Customer")
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "shap_local_explanation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==================================================
# 9. SAVE EXPERIMENT SUMMARY
# ==================================================

summary = {
    "model": "Optimised Logistic Regression",
    "explainer": "SHAP LinearExplainer",
    "background_sample_size": int(background_size),
    "explained_test_sample_size": int(explanation_size),
    "number_of_features": int(len(feature_names)),
    "top_10_features": importance_df.head(10).to_dict(
        orient="records"
    )
}

with open(
    REPORTS_DIR / "shap_explanation_summary.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(summary, file, indent=4)


# ==================================================
# 10. COMPLETION
# ==================================================

print("\n" + "=" * 55)
print("SHAP EXPLAINABILITY COMPLETED SUCCESSFULLY")
print("=" * 55)

print("Feature importance:", REPORTS_DIR / "shap_feature_importance.csv")
print("Summary:", REPORTS_DIR / "shap_explanation_summary.json")
print("Visualisations:", FIGURES_DIR)

