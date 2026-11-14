
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "telco_churn_cleaned.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "monitoring"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_PATH = OUTPUT_DIR / "reference_data.csv"
LABELS_PATH = OUTPUT_DIR / "reference_labels.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# 2. LOAD AND VALIDATE DATA
# ============================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

required_columns = {"customerID", "Churn"}

if not required_columns.issubset(df.columns):
    raise ValueError(
        f"Missing required columns: {required_columns - set(df.columns)}"
    )

if df["customerID"].duplicated().any():
    raise ValueError("Duplicate customer IDs detected.")

y = df["Churn"].map({"No": 0, "Yes": 1})

if y.isna().any():
    raise ValueError("Unexpected values found in Churn target.")

customer_ids = df["customerID"].copy()
X = df.drop(columns=["customerID", "Churn"]).copy()


# ============================================================
# 3. REPRODUCE ORIGINAL TRAIN-TEST SPLIT
# ============================================================

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


# ============================================================
# 4. PREPARE REFERENCE DATA
# ============================================================

numeric_features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

for column in numeric_features:
    X_train[column] = pd.to_numeric(
        X_train[column], errors="raise"
    ).astype("float64")

categorical_features = [
    column for column in X_train.columns
    if column not in numeric_features
]

for column in categorical_features:
    X_train[column] = X_train[column].astype("object")

reference_data = X_train.copy()

reference_labels = pd.DataFrame({
    "customerID": ids_train,
    "actual_churn": y_train.map({
        0: "No",
        1: "Yes"
    })
})


# ============================================================
# 5. VALIDATE AND SAVE
# ============================================================

if len(reference_data) != len(reference_labels):
    raise ValueError("Reference features and labels are misaligned.")

if reference_data.isna().any().any():
    raise ValueError("Missing values detected in reference features.")

if not reference_data.index.equals(reference_labels.index):
    raise ValueError("Reference feature and label indices do not match.")

reference_data.to_csv(REFERENCE_PATH, index=False)
reference_labels.to_csv(LABELS_PATH, index=False)

print("\n" + "=" * 55)
print("CHURNOPS MONITORING REFERENCE DATA PREPARED")
print("=" * 55)

print("Original dataset:", df.shape)
print("Training reference:", reference_data.shape)
print("Held-out test records:", len(X_test))
print("Numerical features:", len(numeric_features))
print("Categorical features:", len(categorical_features))

print("\nReference churn distribution:")
print(reference_labels["actual_churn"].value_counts())

print("\nSaved files:")
print(REFERENCE_PATH)
print(LABELS_PATH)
