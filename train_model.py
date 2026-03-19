"""
train_model.py - Crop Recommendation System
=============================================
This script loads the crop recommendation dataset, preprocesses it,
trains a RandomForestClassifier, evaluates accuracy, saves the model,
and generates visualisation charts (correlation heatmap + feature importance).

Usage:
    python train_model.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (works without a display)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Fix encoding for Windows console
sys.stdout.reconfigure(encoding="utf-8")

# -----------------------------------------------
# 1. Load the dataset
# -----------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Crop_recommendation.csv")
print(f"[INFO] Loading dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# The CSV may contain trailing empty columns (,,) - drop them
df = df.dropna(axis=1, how="all")

# Standardise column names to lowercase & strip whitespace
df.columns = [col.strip().lower() for col in df.columns]

# -----------------------------------------------
# 2. Basic data exploration
# -----------------------------------------------
print("\n[INFO] Dataset shape:", df.shape)
print("\n[INFO] First 5 rows:")
print(df.head())
print("\n[INFO] Column types:")
print(df.dtypes)
print("\n[INFO] Null values per column:")
print(df.isnull().sum())
print("\n[INFO] Unique crops:", df["label"].nunique())
print(df["label"].value_counts())

# -----------------------------------------------
# 3. Prepare features (X) and target (y)
# -----------------------------------------------
# Map CSV column names to expected feature names
FEATURE_COLS = ["nitrogen", "phosphorus", "potassium", "temperature", "humidity", "ph", "rainfall"]
X = df[FEATURE_COLS]
y = df["label"]

print(f"\n[OK] Features shape: {X.shape}")
print(f"[OK] Target shape:   {y.shape}")

# -----------------------------------------------
# 4. Train / Test split (80 / 20)
# -----------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n[INFO] Training samples: {X_train.shape[0]}")
print(f"[INFO] Testing samples:  {X_test.shape[0]}")

# -----------------------------------------------
# 5. Train a RandomForestClassifier
# -----------------------------------------------
print("\n[INFO] Training RandomForestClassifier ...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -----------------------------------------------
# 6. Evaluate the model
# -----------------------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n[RESULT] Model Accuracy: {accuracy * 100:.2f}%")
print("\n[RESULT] Classification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------------------------
# 7. Save the trained model
# -----------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")
joblib.dump(model, MODEL_PATH)
print(f"\n[OK] Model saved to: {MODEL_PATH}")

# -----------------------------------------------
# 8. Visualisation - Correlation Heatmap
# -----------------------------------------------
plt.figure(figsize=(10, 8))
sns.heatmap(
    df[FEATURE_COLS].corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5,
)
plt.title("Feature Correlation Heatmap", fontsize=16)
plt.tight_layout()
HEATMAP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "correlation_heatmap.png")
plt.savefig(HEATMAP_PATH, dpi=150)
plt.close()
print(f"[OK] Correlation heatmap saved to: {HEATMAP_PATH}")

# -----------------------------------------------
# 9. Visualisation - Feature Importance
# -----------------------------------------------
importances = model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
colors = sns.color_palette("viridis", len(FEATURE_COLS))
plt.barh(
    [FEATURE_COLS[i] for i in sorted_idx],
    importances[sorted_idx],
    color=[colors[i] for i in sorted_idx],
)
plt.xlabel("Importance", fontsize=13)
plt.title("Feature Importance (Random Forest)", fontsize=16)
plt.tight_layout()
IMPORTANCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feature_importance.png")
plt.savefig(IMPORTANCE_PATH, dpi=150)
plt.close()
print(f"[OK] Feature importance chart saved to: {IMPORTANCE_PATH}")

print("\n[DONE] Training complete! You can now run the backend and frontend.")
