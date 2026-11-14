
from pathlib import Path
from datetime import datetime, timezone
import json
import re


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DRIFT_REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "monitoring"
    / "data_drift_report.json"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "reports"
    / "monitoring"
    / "monitoring_summary.json"
)

ALERT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "monitoring"
    / "monitoring_alert.json"
)

# Initial operational thresholds.
# These require calibration before production deployment.
FEATURE_WARNING_COUNT = 1
FEATURE_CRITICAL_COUNT = 3

PREDICTION_PSI_WARNING = 0.10
PREDICTION_PSI_CRITICAL = 0.25


# ============================================================
# 2. LOAD EXISTING MONITORING RESULTS
# ============================================================

for path in [DRIFT_REPORT_PATH, SUMMARY_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Required report not found: {path}")

with open(DRIFT_REPORT_PATH, encoding="utf-8") as file:
    drift_report = json.load(file)

with open(SUMMARY_PATH, encoding="utf-8") as file:
    monitoring_summary = json.load(file)


# ============================================================
# 3. EXTRACT FEATURE-LEVEL DRIFT RESULTS
# ============================================================

feature_results = []

for test in drift_report.get("tests", []):

    name = test.get("name", "")

    if not name.startswith("Value Drift for column "):
        continue

    feature = name.replace("Value Drift for column ", "", 1)

    description = test.get("description", "")

    score_match = re.search(
        r"Drift score is ([0-9]+(?:\.[0-9]+)?)",
        description
    )

    threshold_match = re.search(
        r"drift threshold is ([0-9]+(?:\.[0-9]+)?)",
        description
    )

    score = (
        float(score_match.group(1))
        if score_match else None
    )

    threshold = (
        float(threshold_match.group(1))
        if threshold_match else None
    )

    status = test.get("status", "UNKNOWN")

    feature_results.append({
        "feature": feature,
        "drift_score": score,
        "threshold": threshold,
        "status": status,
        "drift_detected": status == "FAIL"
    })


if not feature_results:
    raise ValueError(
        "No feature-level drift results were found."
    )


# ============================================================
# 4. EXTRACT OVERALL DRIFT-SHARE RESULT
# ============================================================

overall_drift_test = None

for test in drift_report.get("tests", []):

    if test.get("name", "").startswith(
        "Share of Drifted Columns"
    ):
        overall_drift_test = {
            "name": test.get("name"),
            "description": test.get("description"),
            "status": test.get("status")
        }
        break


# ============================================================
# 5. EXTRACT PREDICTION DRIFT
# ============================================================

prediction_drift = monitoring_summary.get(
    "prediction_drift", {}
)

prediction_psi = prediction_drift.get("psi")

if prediction_psi is None:
    raise ValueError("Prediction PSI not found in monitoring summary.")


# ============================================================
# 6. CALCULATE MONITORING STATUS
# ============================================================

drifted_features = [
    result
    for result in feature_results
    if result["drift_detected"]
]

drifted_count = len(drifted_features)
total_features = len(feature_results)

drift_share = drifted_count / total_features

status_reasons = []

if drifted_count >= FEATURE_CRITICAL_COUNT:
    status_reasons.append(
        f"{drifted_count} features exceeded their drift thresholds."
    )

if prediction_psi >= PREDICTION_PSI_CRITICAL:
    status_reasons.append(
        f"Prediction PSI ({prediction_psi:.4f}) "
        "reached the critical threshold."
    )

if (
    drifted_count >= FEATURE_CRITICAL_COUNT
    or prediction_psi >= PREDICTION_PSI_CRITICAL
):
    monitoring_status = "CRITICAL"

elif (
    drifted_count >= FEATURE_WARNING_COUNT
    or prediction_psi >= PREDICTION_PSI_WARNING
):
    monitoring_status = "WARNING"

else:
    monitoring_status = "NORMAL"


# ============================================================
# 7. BUILD AUTOMATED MONITORING RECORD
# ============================================================

alert = {
    "project": "ChurnOps",
    "experiment": "Simulated Drift Monitoring",
    "data_status": monitoring_summary.get(
        "data_status",
        "UNKNOWN"
    ),
    "monitoring_timestamp_utc": datetime.now(
        timezone.utc
    ).isoformat(),

    "monitoring_status": monitoring_status,

    "model": {
        "name": monitoring_summary.get("model_name"),
        "version": monitoring_summary.get("model_version")
    },

    "feature_drift": {
        "total_features": total_features,
        "drifted_features": drifted_count,
        "drift_share": round(drift_share, 4),
        "affected_features": drifted_features,
        "overall_evidently_test": overall_drift_test
    },

    "prediction_drift": {
        "psi": prediction_psi,
        "warning_threshold": PREDICTION_PSI_WARNING,
        "critical_threshold": PREDICTION_PSI_CRITICAL
    },

    "alert_reasons": status_reasons,

    "policy": {
        "feature_warning_count": FEATURE_WARNING_COUNT,
        "feature_critical_count": FEATURE_CRITICAL_COUNT,
        "prediction_psi_warning": PREDICTION_PSI_WARNING,
        "prediction_psi_critical": PREDICTION_PSI_CRITICAL,
        "policy_status": "INITIAL - REQUIRES CALIBRATION"
    }
}


# ============================================================
# 8. SAVE ALERT
# ============================================================

with open(ALERT_PATH, "w", encoding="utf-8") as file:
    json.dump(alert, file, indent=4)


# ============================================================
# 9. DISPLAY MONITORING RESULTS
# ============================================================

print("\n" + "=" * 60)
print("CHURNOPS AUTOMATED MONITORING ASSESSMENT")
print("=" * 60)

print("Monitoring status:", monitoring_status)
print("Model:", alert["model"]["name"])
print("Model version:", alert["model"]["version"])

print("\nFeature drift:")
print(f"Drifted features: {drifted_count}/{total_features}")
print(f"Drift share: {drift_share:.2%}")

for feature in drifted_features:
    print(
        f" - {feature['feature']}: "
        f"{feature['drift_score']}"
    )

print("\nPrediction drift:")
print(f"PSI: {prediction_psi:.4f}")

print("\nAlert reasons:")

if status_reasons:
    for reason in status_reasons:
        print("-", reason)
else:
    print("No alert conditions triggered.")

print("\nOverall Evidently drift-share test:")

if overall_drift_test:
    print(overall_drift_test["status"])
    print(overall_drift_test["description"])

print("\nSaved monitoring alert:")
print(ALERT_PATH)
