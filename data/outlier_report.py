import pandas as pd
import numpy as np
import os

# Columns to analyze
diabetes = pd.read_csv('../data/diabetes.csv')
cols = ['Glucose', 'BMI', 'BloodPressure', 'Insulin']

def iqr_outlier_summary(diabetes, columns, save_csv=False, path="data/outlier_summary.csv"):
    summary = []

    for col in columns:
        series = diabetes[col].dropna()

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = ((series < lower) | (series > upper))

        summary.append({
            "column": col,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_bound": lower,
            "upper_bound": upper,
            "outlier_count": outliers.sum(),
            "outlier_pct": outliers.mean() * 100
        })

    summary_df = pd.DataFrame(summary).sort_values(by="outlier_count", ascending=False)

    # Print summary
    print("\nIQR Outlier Summary:\n")
    print(summary_df.to_string(index=False))

    # Optionally save
    if save_csv:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        summary_df.to_csv(path, index=False)
        print(f"\nSaved to {path}")

    return summary_df



outlier_summary = iqr_outlier_summary(df, cols, save_csv=True)