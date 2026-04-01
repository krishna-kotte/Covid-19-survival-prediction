"""
COVID-19 Patient Survival Prediction
Step 1: EDA & Preprocessing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ─── Config ─────────────────────────────────────────────
DATA_PATH = "patient.csv"
OUTPUT_PLOT = "outputs/plots/"
OUTPUT_DATA = "data/"

os.makedirs(OUTPUT_PLOT, exist_ok=True)
os.makedirs(OUTPUT_DATA, exist_ok=True)

# ─── 1. Load Data ───────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading Dataset")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nColumns:\n{df.columns.tolist()}")

# ─── 2. Select Relevant Features ────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Selecting Relevant Features")
print("=" * 60)

FEATURES = [
    "age",
    "sex",
    "diabetes",
    "hypertension",
    "obesity",
    "pneumonia",
    "smoker",
    "another_case",
    "intubated",
    "icu",
    "death_date"
]

available = [c for c in FEATURES if c in df.columns]
df = df[available].copy()

print(f"Columns kept: {df.columns.tolist()}")

# ─── 3. Create Target Variable ──────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Engineering Target Variable (SURVIVED)")
print("=" * 60)

df["SURVIVED"] = df["death_date"].apply(
    lambda x: 0 if str(x) != "9999-99-99" else 1
)

df.drop(columns=["death_date"], inplace=True)

print(f"Survival distribution:\n{df['SURVIVED'].value_counts()}")
print(f"Survival rate: {df['SURVIVED'].mean()*100:.2f}%")

# ─── 4. Handle Missing / Invalid Values ─────────────────
print("\n" + "=" * 60)
print("STEP 4: Handling Missing & Invalid Values")
print("=" * 60)

INVALID = [97, 98, 99]
categorical_cols = [c for c in df.columns if c not in ["age", "SURVIVED"]]

for col in categorical_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].replace(INVALID, np.nan)

print(f"Missing values per column:\n{df.isnull().sum()}")

# Fill missing categorical with mode
for col in categorical_cols:
    mode_val = df[col].mode()[0]
    df[col].fillna(mode_val, inplace=True)

# Fill missing age with median
df["age"].fillna(df["age"].median(), inplace=True)

print(f"\nMissing values after imputation:\n{df.isnull().sum()}")

# ─── 5. EDA Visualizations ──────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: EDA Visualizations")
print("=" * 60)

plt.style.use("seaborn-v0_8-whitegrid")

# Survival distribution
plt.figure(figsize=(6,4))
df["SURVIVED"].value_counts().plot(kind="bar")
plt.title("Survival Distribution")
plt.xlabel("0 = Died, 1 = Survived")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{OUTPUT_PLOT}01_survival_distribution.png")
plt.close()

# Age distribution
plt.figure(figsize=(8,4))
df[df["SURVIVED"] == 1]["age"].plot(kind="hist", bins=30, alpha=0.6, label="Survived")
df[df["SURVIVED"] == 0]["age"].plot(kind="hist", bins=30, alpha=0.6, label="Died")
plt.legend()
plt.title("Age Distribution by Survival")
plt.xlabel("Age")
plt.tight_layout()
plt.savefig(f"{OUTPUT_PLOT}02_age_distribution.png")
plt.close()

# Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="RdYlGn")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{OUTPUT_PLOT}03_correlation_heatmap.png")
plt.close()

# ─── 6. Save Cleaned Data ───────────────────────────────
df.to_csv(f"{OUTPUT_DATA}cleaned_data.csv", index=False)

print(f"\n✔ Cleaned data saved → data/cleaned_data.csv")
print(f"Final dataset shape: {df.shape}")
print("\n✅ EDA & Preprocessing Complete!")