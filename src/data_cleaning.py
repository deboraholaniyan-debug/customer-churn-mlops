
from pathlib import Path
import pandas as pd

# Identify project directories
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = (
    PROJECT_ROOT / "data" / "raw" /
    "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = PROCESSED_DIR / "telco_churn_cleaned.csv"

# Load the raw dataset
print("Loading raw dataset...")
df = pd.read_csv(RAW_DATA_PATH)

print("Original dataset shape:", df.shape)

# Remove leading and trailing spaces from text columns
text_columns = df.select_dtypes(
    include=["object", "string"]
).columns

for column in text_columns:
    df[column] = df[column].astype("string").str.strip()

# Convert blank strings to missing values
df = df.replace(r"^\s*$", pd.NA, regex=True)

# Check duplicate records
print("\nDuplicate rows:", df.duplicated().sum())
print("Duplicate customer IDs:", df["customerID"].duplicated().sum())

if df.duplicated().any():
    raise ValueError("Duplicate rows detected.")

if df["customerID"].duplicated().any():
    raise ValueError("Duplicate customer IDs detected.")

# Convert numerical columns
numeric_columns = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Examine missing TotalCharges values
missing_before = df["TotalCharges"].isna().sum()

print("\nMissing TotalCharges before treatment:", missing_before)

affected_records = df.loc[
    df["TotalCharges"].isna(),
    ["customerID", "tenure", "MonthlyCharges"]
]

print("\nAffected records:")
print(affected_records)

# Replace missing TotalCharges with zero only for zero-tenure customers
zero_tenure_missing = (
    df["TotalCharges"].isna() & (df["tenure"] == 0)
)

df.loc[zero_tenure_missing, "TotalCharges"] = 0

# Validate remaining missing values
remaining_missing = df["TotalCharges"].isna().sum()

if remaining_missing > 0:
    raise ValueError(
        f"{remaining_missing} unresolved TotalCharges values remain."
    )

# Validate important fields
if not df["SeniorCitizen"].isin([0, 1]).all():
    raise ValueError("Invalid SeniorCitizen values.")

if not df["Churn"].isin(["Yes", "No"]).all():
    raise ValueError("Invalid Churn values.")

if (df["tenure"] < 0).any() or (df["tenure"] > 72).any():
    raise ValueError("Invalid tenure values.")

if (df["MonthlyCharges"] < 0).any():
    raise ValueError("Negative MonthlyCharges detected.")

if (df["TotalCharges"] < 0).any():
    raise ValueError("Negative TotalCharges detected.")

if df.isnull().any().any():
    raise ValueError("Missing values remain in the dataset.")

# Save cleaned dataset
df.to_csv(OUTPUT_PATH, index=False)

# Display cleaning summary
print("\n" + "=" * 50)
print("DATA CLEANING COMPLETED")
print("=" * 50)

print("Original records:", len(df))
print("Cleaned dataset shape:", df.shape)
print("Missing values:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())
print("Unique customers:", df["customerID"].nunique())
print("TotalCharges type:", df["TotalCharges"].dtype)
print("Zero-tenure customers:", (df["tenure"] == 0).sum())
print("Missing values treated:", missing_before)

print("\nSaved cleaned dataset to:")
print(OUTPUT_PATH)
