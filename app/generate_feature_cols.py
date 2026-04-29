"""
generate_feature_cols.py
────────────────────────
Run this once to generate models/feature_cols.pkl
which is required by the Streamlit app.

Usage:
    python3 generate_feature_cols.py
"""

import joblib
import os

FEATURE_COLS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "BMI_Category",
    "BP_Range",
    "Glucose_Insulin_Ratio",
]

output_path = os.path.join(os.path.dirname(__file__), "..", "models", "feature_cols.pkl")
output_path = os.path.normpath(output_path)

joblib.dump(FEATURE_COLS, output_path)
print(f"✅ Saved feature_cols.pkl to: {output_path}")
print(f"   Columns: {FEATURE_COLS}")
