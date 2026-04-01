"""
COVID-19 Patient Survival Prediction
Step 3: Predict on New / Unseen Patient Data
"""

import pandas as pd
import joblib
import os
import glob

# ─── Load Best Saved Model ─────────────────────────────
MODEL_DIR = "outputs/models/"
model_files = glob.glob(f"{MODEL_DIR}*.pkl")

if not model_files:
    print("❌ No trained model found. Run 02_model_training.py first.")
    exit()

model_path = model_files[0]
model = joblib.load(model_path)
print(f"✔ Loaded model: {model_path}")

# ─── Define Patient Features (Must match training columns) ─────────────
# Encoding: 1 = Yes, 2 = No

new_patient = pd.DataFrame([{
    "age": 65,
    "sex": 1,
    "diabetes": 1,
    "hypertension": 1,
    "obesity": 2,
    "pneumonia": 1,
    "smoker": 2,
    "another_case": 1,
    "intubated": 2,
    "icu": 2
}])

# ─── Ensure correct column order ───────────────────────
try:
    feature_names = model.get_booster().feature_names  # XGBoost
except AttributeError:
    feature_names = model.feature_names_in_             # sklearn models

new_patient = new_patient[feature_names]

# ─── Predict ──────────────────────────────────────────
prediction = model.predict(new_patient)[0]
probability = model.predict_proba(new_patient)[0]

print("\n" + "=" * 50)
print("PATIENT SURVIVAL PREDICTION")
print("=" * 50)
print(f"Survival Probability : {probability[1]*100:.2f}%")
print(f"Death Probability    : {probability[0]*100:.2f}%")
print(f"Prediction           : {'SURVIVED' if prediction == 1 else 'HIGH RISK'}")
print("=" * 50)

# ─── Batch Prediction from CSV ─────────────────────────
BATCH_FILE = "data/new_patients.csv"

if os.path.exists(BATCH_FILE):
    print(f"\nRunning batch predictions on: {BATCH_FILE}")
    batch_df = pd.read_csv(BATCH_FILE)
    batch_df = batch_df[feature_names]

    batch_preds = model.predict(batch_df)
    batch_probs = model.predict_proba(batch_df)[:, 1]

    batch_df["SURVIVED_PRED"] = batch_preds
    batch_df["SURVIVAL_PROB"] = (batch_probs * 100).round(2)
    batch_df["RISK_LEVEL"] = batch_df["SURVIVAL_PROB"].apply(
        lambda p: "LOW RISK" if p >= 70 else ("MODERATE" if p >= 40 else "HIGH RISK")
    )

    os.makedirs("outputs/reports/", exist_ok=True)
    batch_df.to_csv("outputs/reports/batch_predictions.csv", index=False)
    print("✔ Batch predictions saved → outputs/reports/batch_predictions.csv")
    print(batch_df[["SURVIVED_PRED", "SURVIVAL_PROB", "RISK_LEVEL"]].head())