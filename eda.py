from pathlib import Path
import pandas as pd

# eda.py is located in the project root
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "telco_churn_cleaned.csv"
)

print("Project root:", PROJECT_ROOT)
print("Dataset path:", DATA_PATH)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Cleaned dataset not found: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)
# # Display dataset information
# print("Dataset successfully loaded!")
# print("\nFirst five records:")
# print(df.head())

# print("\nDataset dimensions:")
# print(df.shape)

# print("\nColumn names:")
# print(df.columns.tolist())

# print("\nDataset information:")
# print(df.info())

# print("\nMissing values:")
# print(df.isnull().sum())

# print("\ntarget distribution:")
# print(df["Churn"].value_counts())

# print("\nStatistical summary:")
# print(df.describe())

# from pathlib import Path

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns


# # ==========================================
# # 1. PROJECT CONFIGURATION
# # ==========================================

# PROJECT_ROOT = Path(__file__).resolve().parents[1]

# DATA_PATH = (
#     PROJECT_ROOT
#     / "data"
#     / "raw"
#     / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
# )

# OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# sns.set_theme(style="whitegrid", palette="deep")

# # Clean TotalCharges for analysis
# df["TotalCharges"] = pd.to_numeric(
#     df["TotalCharges"].astype(str).str.strip(),
#     errors="coerce"
# )

# # Remove customer identifier from analysis features
# analysis_df = df.drop(columns=["customerID"])

# print("Dataset loaded:", df.shape)


# # ==========================================
# # 2. HELPER FUNCTION TO SAVE PLOTS
# # ==========================================

# def save_plot(filename):
#     plt.tight_layout()
#     plt.savefig(
#         OUTPUT_DIR / filename,
#         dpi=300,
#         bbox_inches="tight"
#     )
#     plt.show()
#     plt.close()


# # ==========================================
# # 3. CHURN DISTRIBUTION
# # ==========================================

# plt.figure(figsize=(8, 5))

# ax = sns.countplot(
#     data=df,
#     x="Churn",
#     hue="Churn",
#     order=["No", "Yes"],
#     legend=False
# )

# plt.title("Customer Churn Distribution")
# plt.xlabel("Churn Status")
# plt.ylabel("Number of Customers")

# for container in ax.containers:
#     ax.bar_label(container)

# save_plot("01_churn_distribution.png")


# # ==========================================
# # 4. CHURN PERCENTAGE
# # ==========================================

# churn_percentage = (
#     df["Churn"]
#     .value_counts(normalize=True)
#     .reindex(["No", "Yes"])
#     * 100
# )

# plt.figure(figsize=(7, 7))

# plt.pie(
#     churn_percentage,
#     labels=["Retained", "Churned"],
#     autopct="%1.1f%%",
#     startangle=90,
#     explode=(0, 0.08)
# )

# plt.title("Percentage of Customers Who Churned")

# save_plot("02_churn_percentage.png")


# # ==========================================
# # 5. NUMERICAL FEATURE DISTRIBUTIONS
# # ==========================================

# numerical_features = [
#     "tenure",
#     "MonthlyCharges",
#     "TotalCharges"
# ]

# fig, axes = plt.subplots(
#     1, 3,
#     figsize=(17, 5)
# )

# for ax, column in zip(axes, numerical_features):
#     sns.histplot(
#         data=df,
#         x=column,
#         kde=True,
#         bins=30,
#         ax=ax
#     )

#     ax.set_title(f"Distribution of {column}")

# plt.suptitle("Numerical Feature Distributions")

# plt.tight_layout()

# plt.savefig(
#     OUTPUT_DIR / "03_numerical_distributions.png",
#     dpi=300,
#     bbox_inches="tight"
# )

# plt.show()
# plt.close()


# # ==========================================
# # 6. TENURE VS CHURN
# # ==========================================

# plt.figure(figsize=(9, 5))

# sns.boxplot(
#     data=df,
#     x="Churn",
#     y="tenure",
#     hue="Churn",
#     order=["No", "Yes"],
#     legend=False
# )

# plt.title("Customer Tenure by Churn Status")
# plt.xlabel("Churn Status")
# plt.ylabel("Tenure (Months)")

# save_plot("04_tenure_vs_churn.png")


# # ==========================================
# # 7. MONTHLY CHARGES VS CHURN
# # ==========================================

# plt.figure(figsize=(9, 5))

# sns.boxplot(
#     data=df,
#     x="Churn",
#     y="MonthlyCharges",
#     hue="Churn",
#     order=["No", "Yes"],
#     legend=False
# )

# plt.title("Monthly Charges by Churn Status")
# plt.xlabel("Churn Status")
# plt.ylabel("Monthly Charges")

# save_plot("05_monthly_charges_vs_churn.png")


# # ==========================================
# # 8. CHURN BY CONTRACT TYPE
# # ==========================================

# plt.figure(figsize=(9, 5))

# sns.countplot(
#     data=df,
#     x="Contract",
#     hue="Churn",
#     order=[
#         "Month-to-month",
#         "One year",
#         "Two year"
#     ]
# )

# plt.title("Customer Churn by Contract Type")
# plt.xlabel("Contract Type")
# plt.ylabel("Number of Customers")
# plt.legend(title="Churn")

# save_plot("06_churn_by_contract.png")


# # ==========================================
# # 9. CHURN BY PAYMENT METHOD
# # ==========================================

# plt.figure(figsize=(11, 5))

# sns.countplot(
#     data=df,
#     x="PaymentMethod",
#     hue="Churn"
# )

# plt.title("Customer Churn by Payment Method")
# plt.xlabel("Payment Method")
# plt.ylabel("Number of Customers")
# plt.xticks(rotation=25, ha="right")
# plt.legend(title="Churn")

# save_plot("07_churn_by_payment_method.png")


# # ==========================================
# # 10. CHURN RATE BY INTERNET SERVICE
# # ==========================================

# internet_churn = (
#     df.groupby("InternetService", observed=False)["Churn"]
#     .apply(lambda x: (x == "Yes").mean() * 100)
#     .sort_values(ascending=False)
# )

# plt.figure(figsize=(9, 5))

# sns.barplot(
#     x=internet_churn.index,
#     y=internet_churn.values
# )

# plt.title("Churn Rate by Internet Service")
# plt.xlabel("Internet Service")
# plt.ylabel("Churn Rate (%)")

# save_plot("08_churn_rate_by_internet_service.png")


# # ==========================================
# # 11. CORRELATION HEATMAP
# # ==========================================

# correlation_df = df.copy()

# correlation_df["Churn_binary"] = (
#     correlation_df["Churn"].map({"No": 0, "Yes": 1})
# )

# correlation_features = [
#     "SeniorCitizen",
#     "tenure",
#     "MonthlyCharges",
#     "TotalCharges",
#     "Churn_binary"
# ]

# correlation_matrix = correlation_df[
#     correlation_features
# ].corr()

# plt.figure(figsize=(9, 6))

# sns.heatmap(
#     correlation_matrix,
#     annot=True,
#     cmap="coolwarm",
#     fmt=".2f",
#     linewidths=0.5
# )

# plt.title("Correlation Between Numerical Features and Churn")

# save_plot("09_correlation_heatmap.png")


# # ==========================================
# # 12. CHURN RATE BY TENURE GROUP
# # ==========================================

# df["TenureGroup"] = pd.cut(
#     df["tenure"],
#     bins=[-1, 12, 24, 48, 60, 72],
#     labels=[
#         "0-12 months",
#         "13-24 months",
#         "25-48 months",
#         "49-60 months",
#         "61-72 months"
#     ]
# )

# tenure_churn = (
#     df.groupby("TenureGroup", observed=False)["Churn"]
#     .apply(lambda x: (x == "Yes").mean() * 100)
# )

# plt.figure(figsize=(10, 5))

# sns.barplot(
#     x=tenure_churn.index,
#     y=tenure_churn.values
# )

# plt.title("Churn Rate Across Customer Tenure Groups")
# plt.xlabel("Tenure Group")
# plt.ylabel("Churn Rate (%)")
# plt.xticks(rotation=15)

# save_plot("10_churn_by_tenure_group.png")


# print("\nEDA visualisations completed successfully.")
# print("Saved figures to:", OUTPUT_DIR)


