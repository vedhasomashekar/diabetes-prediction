"""
Model Training & Evaluation Module
- Trains Logistic Regression (baseline) and Random Forest Classifier
- Evaluates on validation set using accuracy, precision, recall, F1
- Saves the best model for use in the Streamlit app

Usage:
    python models/train.py

Prerequisites:
    Run data/preprocess.py first to generate the cleaned datasets.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import joblib
import os
import json

def load_processed_data():
    """Load the preprocessed train/val/test splits."""
    X_train = pd.read_csv("data/X_train.csv")
    X_val = pd.read_csv("data/X_val.csv")
    X_test = pd.read_csv("data/X_test.csv")
    y_train = pd.read_csv("data/y_train.csv").values.ravel()
    y_val = pd.read_csv("data/y_val.csv").values.ravel()
    y_test = pd.read_csv("data/y_test.csv").values.ravel()

    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test

def evaluate_model(model, X, y, dataset_name="Validation"):
    """Evaluate a model and return metrics dict."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None

    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred),
    }

    print(f"\n--- {dataset_name} Results ---")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"\n  Classification Report:\n{classification_report(y, y_pred)}")
    print(f"  Confusion Matrix:\n{confusion_matrix(y, y_pred)}")

    return metrics

def train_logistic_regression(X_train, y_train):
    """Train a Logistic Regression baseline model."""
    print("\n" + "=" * 50)
    print("Training: Logistic Regression")
    print("=" * 50)

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        C=1.0,
        solver='lbfgs'
    )
    model.fit(X_train, y_train)
    print("Training complete.")
    return model

def train_random_forest(X_train, y_train):
    """Train a Random Forest Classifier."""
    print("\n" + "=" * 50)
    print("Training: Random Forest Classifier")
    print("=" * 50)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("Training complete.")
    return model

def main():
    os.makedirs("models", exist_ok=True)

    # Load data
    print("Loading preprocessed data...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data()

    # Train models
    lr_model = train_logistic_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    # Evaluate on validation set
    print("\n" + "=" * 50)
    print("VALIDATION SET EVALUATION")
    print("=" * 50)

    lr_metrics = evaluate_model(lr_model, X_val, y_val, "Logistic Regression - Validation")
    rf_metrics = evaluate_model(rf_model, X_val, y_val, "Random Forest - Validation")

    # Pick the best model by F1 score
    if rf_metrics['f1'] >= lr_metrics['f1']:
        best_model = rf_model
        best_name = "Random Forest"
        best_metrics = rf_metrics
    else:
        best_model = lr_model
        best_name = "Logistic Regression"
        best_metrics = lr_metrics

    print("\n" + "=" * 50)
    print(f"BEST MODEL: {best_name} (F1: {best_metrics['f1']:.4f})")
    print("=" * 50)

    # Evaluate best model on test set
    test_metrics = evaluate_model(best_model, X_test, y_test, f"{best_name} - Test Set")

    # Save the best model
    joblib.dump(best_model, "models/best_model.pkl")
    joblib.dump(lr_model, "models/logistic_regression.pkl")
    joblib.dump(rf_model, "models/random_forest.pkl")
    joblib.dump(list(X_train.columns), "models/feature_cols.pkl")

    # Save metrics for the app to display
    results = {
        'best_model': best_name,
        'logistic_regression': {k: round(v, 4) for k, v in lr_metrics.items()},
        'random_forest': {k: round(v, 4) for k, v in rf_metrics.items()},
        'test_set': {k: round(v, 4) for k, v in test_metrics.items()},
    }
    with open("models/metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved:")
    print("  models/best_model.pkl")
    print("  models/logistic_regression.pkl")
    print("  models/random_forest.pkl")
    print("  models/feature_cols.pkl")
    print("  models/metrics.json")

if __name__ == "__main__":
    main()