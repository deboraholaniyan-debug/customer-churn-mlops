
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REFERENCE_PATH = (
    PROJECT_ROOT / "data" / "monitoring" / "reference_data.csv"
)

MODEL_PATH = PROJECT_ROOT / "deployment" / "model.joblib"

METADATA_PATH = (
    PROJECT_ROOT / "deployment" / "model_metadata.json"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "monitoring"
SIMULATED_DIR = PROJECT_ROOT / "data" / "simulated"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SIMULATED_DIR.mkdir(parents=True, exist_ok=True)

SIMULATED_DATA_PATH = (
    SIMULATED_DIR / "simulated_production_batch_001.csv"
)

PREDICTIONS_PATH = (
    OUTPUT_DIR / "prediction_drift_comparison.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR / "monitoring_summary.json"
)

HTML_REPORT_PATH = (
    OUTPUT_DIR / "data_drift_report.html"
)

JSON_REPORT_PATH = (
    OUTPUT_DIR / "data_drift_report.json"
)

RANDOM_STATE = 42
CURRENT_BATCH_SIZE = 1500


# ============================================================
# 2. LOAD REFERENCE DATA AND MODEL
# ============================================================

if not REFERENCE_PATH.exists():
    raise FileNotFoundError(
        f"Reference dataset not found: {REFERENCE_PATH}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

reference_data = pd.read_csv(REFERENCE_PATH)

model = joblib.load(MODEL_PATH)

with open(METADATA_PATH, "r", encoding="utf-8") as file:
    metadata = json.load(file)

threshold = float(metadata["classification_threshold"])

print("Reference records:", len(reference_data))
print("Reference features:", len(reference_data.columns))
print("Model:", metadata["model_name"])
print("Model version:", metadata["model_version"])
print("Classification threshold:", threshold)


# ============================================================
# 3. GENERATE SIMULATED PRODUCTION BATCH
# ============================================================

# Weighted resampling preserves realistic combinations of
# customer characteristics while changing segment representation.

sampling_weights = pd.Series(
    np.ones(len(reference_data)),
    index=reference_data.index,
    dtype="float64"
)

# Increase representation of newer customers.
sampling_weights *= np.where(
    reference_data["tenure"] <= 12,
    2.5,
    1.0
)

# Increase representation of month-to-month customers.
sampling_weights *= np.where(
    reference_data["Contract"] == "Month-to-month",
    1.8,
    1.0
)

# Increase representation of electronic-check customers.
sampling_weights *= np.where(
    reference_data["PaymentMethod"] == "Electronic check",
    1.7,
    1.0
)

sampling_weights /= sampling_weights.sum()

simulated_batch = reference_data.sample(
    n=CURRENT_BATCH_SIZE,
    replace=False,
    weights=sampling_weights,
    random_state=RANDOM_STATE
).reset_index(drop=True)

simulated_batch.to_csv(
    SIMULATED_DATA_PATH,
    index=False
)

print("\nSimulated production batch created.")
print("Batch size:", len(simulated_batch))
print("Saved to:", SIMULATED_DATA_PATH)


# ============================================================
# 4. GENERATE DATA DRIFT REPORT
# ============================================================

drift_report = Report(
    metrics=[
        DataDriftPreset(
            columns=list(reference_data.columns),
            drift_share=0.5
        )
    ],
    include_tests=True
)

snapshot = drift_report.run(
    reference_data=reference_data,
    current_data=simulated_batch
)

snapshot.save_html(str(HTML_REPORT_PATH))
snapshot.save_json(str(JSON_REPORT_PATH))

print("\nEvidently data drift report generated.")
print("HTML report:", HTML_REPORT_PATH)
print("JSON report:", JSON_REPORT_PATH)


# ============================================================
# 5. GENERATE MODEL PREDICTIONS
# ============================================================

reference_probabilities = model.predict_proba(
    reference_data
)[:, 1]

current_probabilities = model.predict_proba(
    simulated_batch
)[:, 1]

reference_predictions = (
    reference_probabilities >= threshold
).astype(int)

current_predictions = (
    current_probabilities >= threshold
).astype(int)


# ============================================================
# 6. CALCULATE PREDICTION DRIFT
# ============================================================

def calculate_psi(reference, current, bins=10):
    """
    Calculate Population Stability Index (PSI).

    Fixed probability bins make the comparison consistent
    between reference and simulated production predictions.
    """

    edges = np.linspace(0.0, 1.0, bins + 1)

    reference_counts, _ = np.histogram(
        reference,
        bins=edges
    )

    current_counts, _ = np.histogram(
        current,
        bins=edges
    )

    epsilon = 1e-6

    reference_share = (
        reference_counts + epsilon
    ) / (len(reference) + bins * epsilon)

    current_share = (
        current_counts + epsilon
    ) / (len(current) + bins * epsilon)

    psi = np.sum(
        (current_share - reference_share)
        * np.log(current_share / reference_share)
    )

    return float(psi)


prediction_psi = calculate_psi(
    reference_probabilities,
    current_probabilities
)

reference_churn_rate = float(
    reference_predictions.mean() * 100
)

current_churn_rate = float(
    current_predictions.mean() * 100
)

reference_mean_probability = float(
    reference_probabilities.mean()
)

current_mean_probability = float(
    current_probabilities.mean()
)

prediction_comparison = pd.DataFrame({
    "dataset": (
        ["reference"] * len(reference_probabilities)
        + ["simulated_production"] * len(current_probabilities)
    ),
    "churn_probability": np.concatenate([
        reference_probabilities,
        current_probabilities
    ]),
    "predicted_churn": np.concatenate([
        reference_predictions,
        current_predictions
    ])
})

prediction_comparison.to_csv(
    PREDICTIONS_PATH,
    index=False
)


# ============================================================
# 7. BUILD MONITORING SUMMARY
# ============================================================

summary = {
    "experiment": "ChurnOps Simulated Drift Monitoring",
    "data_status": "SIMULATED - NOT REAL PRODUCTION DATA",
    "random_state": RANDOM_STATE,
    "reference_records": len(reference_data),
    "simulated_records": len(simulated_batch),
    "model_name": metadata["model_name"],
    "model_version": metadata["model_version"],
    "classification_threshold": threshold,
    "prediction_drift": {
        "psi": prediction_psi,
        "reference_mean_churn_probability":
            reference_mean_probability,
        "simulated_mean_churn_probability":
            current_mean_probability,
        "reference_predicted_churn_rate_percent":
            reference_churn_rate,
        "simulated_predicted_churn_rate_percent":
            current_churn_rate,
        "predicted_churn_rate_change_percentage_points":
            current_churn_rate - reference_churn_rate
    },
    "limitations": [
        "The current batch is simulated using weighted resampling.",
        "Detected drift does not establish model performance degradation.",
        "Actual production labels are required for outcome-based evaluation.",
        "Prediction drift is not evidence of causal change."
    ]
}

with open(SUMMARY_PATH, "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=4)

print("\n" + "=" * 60)
print("CHURNOPS DRIFT MONITORING COMPLETED")
print("=" * 60)

print("\nPrediction drift summary:")
print(f"Prediction PSI: {prediction_psi:.4f}")

print(
    "Reference mean churn probability:",
    f"{reference_mean_probability:.4f}"
)

print(
    "Simulated mean churn probability:",
    f"{current_mean_probability:.4f}"
)

print(
    "Reference predicted churn rate:",
    f"{reference_churn_rate:.2f}%"
)

print(
    "Simulated predicted churn rate:",
    f"{current_churn_rate:.2f}%"
)

print("\nGenerated artefacts:")
print(HTML_REPORT_PATH)
print(JSON_REPORT_PATH)
print(PREDICTIONS_PATH)
print(SUMMARY_PATH)
