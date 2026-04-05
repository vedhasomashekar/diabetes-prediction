"""
Data Preprocessing Module
- Loads the Pima Indians Diabetes Dataset
- Handles missing/invalid zero values via median imputation
- Engineers new features (BMI category, BP range)
- Normalizes numerical features
- Splits into train/val/test (70/15/15)
- Saves cleaned datasets to data/

Usage:
    python data/preprocess.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import joblib

def load_data(filepath="data/diabetes.csv"):
    """Load the raw dataset."""
    df = pd.read_csv(filepath)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Class distribution:\n{df['Outcome'].value_counts()}")
    return df

def handle_missing_values(df):
    """
    Replace biologically implausible zero values with median.
    Columns where 0 is invalid: Glucose, BloodPressure, SkinThickness,
    Insulin, BMI
    """
    zero_invalid_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

    for col in zero_invalid_cols:
        zero_count = (df[col] == 0).sum()
        zero_pct = zero_count / len(df) * 100
        print(f"  {col}: {zero_count} zeros ({zero_pct:.1f}%)")

        median_val = df[col][df[col] != 0].median()
        df[col] = df[col].replace(0, median_val)

    print("Zero values replaced with median imputation.")
    return df

def engineer_features(df):
    """Create derived features for better predictive power."""

    # BMI categories (WHO classification)
    def bmi_category(bmi):
        if bmi < 18.5:
            return 0  # Underweight
        elif bmi < 25:
            return 1  # Normal
        elif bmi < 30:
            return 2  # Overweight
        else:
            return 3  # Obese

    df['BMI_Category'] = df['BMI'].apply(bmi_category)

    # Blood pressure ranges
    def bp_range(bp):
        if bp < 80:
            return 0  # Normal
        elif bp < 90:
            return 1  # Elevated
        else:
            return 2  # High

    df['BP_Range'] = df['BloodPressure'].apply(bp_range)

    # Glucose-to-Insulin ratio (can indicate insulin resistance)
    df['Glucose_Insulin_Ratio'] = df['Glucose'] / (df['Insulin'] + 1)

    print(f"Engineered 3 new features. Total features: {df.shape[1]}")
    return df

def split_and_scale(df, target_col='Outcome', test_size=0.15, val_size=0.15):
    """
    Split into train/val/test (70/15/15) and apply StandardScaler.
    Scaler is fit ONLY on training data to prevent data leakage.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Second split: separate validation from training
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
    )

    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    # Scale features - fit on train only
    scaler = StandardScaler()
    feature_cols = X_train.columns.tolist()

    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=feature_cols, index=X_val.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler, feature_cols

def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Step 1: Load
    print("=" * 50)
    print("STEP 1: Loading data")
    print("=" * 50)
    df = load_data()

    # Step 2: Handle missing values
    print("\n" + "=" * 50)
    print("STEP 2: Handling missing values")
    print("=" * 50)
    df = handle_missing_values(df)

    # Step 3: Feature engineering
    print("\n" + "=" * 50)
    print("STEP 3: Feature engineering")
    print("=" * 50)
    df = engineer_features(df)

    # Step 4: Split and scale
    print("\n" + "=" * 50)
    print("STEP 4: Train/Val/Test split & scaling")
    print("=" * 50)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_cols = split_and_scale(df)

    # Step 5: Save everything
    print("\n" + "=" * 50)
    print("STEP 5: Saving processed data")
    print("=" * 50)

    X_train.to_csv("data/X_train.csv", index=False)
    X_val.to_csv("data/X_val.csv", index=False)
    X_test.to_csv("data/X_test.csv", index=False)
    y_train.to_csv("data/y_train.csv", index=False)
    y_val.to_csv("data/y_val.csv", index=False)
    y_test.to_csv("data/y_test.csv", index=False)
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(feature_cols, "models/feature_cols.pkl")

    print("All files saved successfully!")
    print("  data/X_train.csv, X_val.csv, X_test.csv")
    print("  data/y_train.csv, y_val.csv, y_test.csv")
    print("  models/scaler.pkl, models/feature_cols.pkl")

if __name__ == "__main__":
    main()
