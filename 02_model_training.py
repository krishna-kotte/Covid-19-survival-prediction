"""
COVID-19 Patient Survival Prediction
Step 2: Model Training — XGBoost vs Random Forest
Optimizing RECALL to minimize false negatives
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, recall_score,
    precision_score, f1_score, accuracy_score
)
from xgboost import XGBClassifier

# ─── Config ───────────────────────────────────────────────────────────────────
DATA_PATH    = "data/cleaned_data.csv"
OUTPUT_MODEL = "outputs/models/"
OUTPUT_PLOT  = "outputs/plots/"
RANDOM_STATE = 42

os.makedirs(OUTPUT_MODEL, exist_ok=True)
os.makedirs(OUTPUT_PLOT,  exist_ok=True)

# ─── 1. Load Cleaned Data ─────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading Cleaned Data")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape}")
print(f"Target distribution:\n{df['SURVIVED'].value_counts()}")

# ─── 2. Train-Test Split ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Splitting Data")
print("=" * 60)

X = df.drop(columns=["SURVIVED"])
y = df["SURVIVED"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
print(f"Train survival rate: {y_train.mean()*100:.2f}%")
print(f"Test  survival rate: {y_test.mean()*100:.2f}%")

# ─── 3. Calculate Class Weight (handles imbalance) ────────────────────────────
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos   # for XGBoost
print(f"\nClass weight ratio (neg/pos): {scale_pos_weight:.2f}")

# ─── 4. Define Models ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Training Models")
print("=" * 60)

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",   # handles imbalance
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,  # handles imbalance
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0
    )
}

results = {}
trained_models = {}

for name, model in models.items():
    print(f"\n{'─'*40}")
    print(f"Training: {name}")
    print(f"{'─'*40}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy" : accuracy_score(y_test, y_pred),
        "Recall"   : recall_score(y_test, y_pred),       # ← Key metric
        "Precision": precision_score(y_test, y_pred),
        "F1 Score" : f1_score(y_test, y_pred),
        "ROC-AUC"  : roc_auc_score(y_test, y_prob),
    }

    results[name] = metrics
    trained_models[name] = (model, y_pred, y_prob)

    print(f"  Accuracy  : {metrics['Accuracy']:.4f}")
    print(f"  Recall    : {metrics['Recall']:.4f}  ← Minimize false negatives")
    print(f"  Precision : {metrics['Precision']:.4f}")
    print(f"  F1 Score  : {metrics['F1 Score']:.4f}")
    print(f"  ROC-AUC   : {metrics['ROC-AUC']:.4f}")

    print(f"\nClassification Report ({name}):")
    print(classification_report(y_test, y_pred, target_names=["Died", "Survived"]))

# ─── 5. Select Best Model (by Recall) ────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Model Selection (by Recall)")
print("=" * 60)

best_name = max(results, key=lambda k: results[k]["Recall"])
print(f"✅ Best model by Recall: {best_name}")
print(f"   Recall = {results[best_name]['Recall']:.4f}")

best_model, best_pred, best_prob = trained_models[best_name]
joblib.dump(best_model, f"{OUTPUT_MODEL}best_model_{best_name.replace(' ','_')}.pkl")
print(f"✔ Model saved → outputs/models/")

# ─── 6. Comparison Table ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Model Comparison Summary")
print("=" * 60)
results_df = pd.DataFrame(results).T.round(4)
print(results_df)

# ─── 7. Plots ─────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")

# (a) Metrics comparison bar chart
fig, ax = plt.subplots(figsize=(9, 5))
results_df.plot(kind="bar", ax=ax, edgecolor="black", width=0.6)
ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
ax.set_xticklabels(list(results.keys()), rotation=0)
ax.set_ylim(0, 1.1)
ax.set_ylabel("Score")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUTPUT_PLOT}05_model_comparison.png", dpi=150)
plt.close()
print("✔ Saved: 05_model_comparison.png")

# (b) Confusion matrices side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (name, (model, y_pred, y_prob)) in zip(axes, trained_models.items()):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Died", "Survived"],
                yticklabels=["Died", "Survived"])
    ax.set_title(f"{name}\nConfusion Matrix", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{OUTPUT_PLOT}06_confusion_matrices.png", dpi=150)
plt.close()
print("✔ Saved: 06_confusion_matrices.png")

# (c) ROC curves
fig, ax = plt.subplots(figsize=(7, 6))
colors = ["#3498db", "#e74c3c"]
for (name, (model, y_pred, y_prob)), color in zip(trained_models.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", color=color, lw=2)
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate (Recall)")
ax.set_title("ROC Curves", fontsize=14, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_PLOT}07_roc_curves.png", dpi=150)
plt.close()
print("✔ Saved: 07_roc_curves.png")

# (d) Feature Importance (XGBoost)
if "XGBoost" in trained_models:
    xgb_model = trained_models["XGBoost"][0]
    importances = pd.Series(xgb_model.feature_importances_, index=X.columns).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    importances.plot(kind="barh", ax=ax, color="#3498db", edgecolor="black")
    ax.set_title("XGBoost — Feature Importances", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PLOT}08_feature_importance.png", dpi=150)
    plt.close()
    print("✔ Saved: 08_feature_importance.png")

print("\n✅ Model Training Complete!")
print(f"   Best Model : {best_name}")
print(f"   Recall     : {results[best_name]['Recall']:.4f}")
print(f"   ROC-AUC    : {results[best_name]['ROC-AUC']:.4f}")
