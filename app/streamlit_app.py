"""
Diabetes Risk Prediction App
CS 6440 - Health Informatics Practicum

A Streamlit web application that uses trained ML models to predict
diabetes risk based on patient health metrics.

Usage:
    streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# ─── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
)

# ─── Load Model & Scaler ────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    feature_cols = joblib.load("models/feature_cols.pkl")
    with open("models/metrics.json", "r") as f:
        metrics = json.load(f)
    return model, scaler, feature_cols, metrics

# ─── Sidebar ────────────────────────────────────────────────
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Predict", "Model Performance", "About"])

# ─── Prediction Page ─────────────────────────────────────────
if page == "Predict":
    st.title("🩺 Diabetes Risk Prediction")
    st.markdown("Enter patient health metrics below to assess diabetes risk.")
    st.markdown("---")

    try:
        model, scaler, feature_cols, metrics = load_model()
        model_loaded = True
    except Exception as e:
        st.error(f"Model not found. Please run preprocessing and training first. Error: {e}")
        model_loaded = False

    # Input form
    col1, col2, col3 = st.columns(3)

    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1,
                                       help="Number of times pregnant")
        glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=300, value=120,
                                   help="Plasma glucose concentration (2hr oral glucose tolerance test)")
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70,
                                          help="Diastolic blood pressure")

    with col2:
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20,
                                          help="Triceps skin fold thickness")
        insulin = st.number_input("Insulin (mu U/ml)", min_value=0, max_value=900, value=80,
                                   help="2-Hour serum insulin")
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1,
                               help="Body mass index (weight in kg / height in m^2)")

    with col3:
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, step=0.01,
                               help="Diabetes pedigree function (genetic risk score)")
        age = st.number_input("Age", min_value=1, max_value=120, value=30,
                               help="Age in years")

    st.markdown("---")

    if st.button("🔍 Predict Risk", type="primary", use_container_width=True):
        if model_loaded:
            # Build input in the same order as training features
            raw_input = {
                'Pregnancies': pregnancies,
                'Glucose': glucose,
                'BloodPressure': blood_pressure,
                'SkinThickness': skin_thickness,
                'Insulin': insulin,
                'BMI': bmi,
                'DiabetesPedigreeFunction': dpf,
                'Age': age,
            }

            # Engineer the same derived features as in preprocessing
            def bmi_category(bmi_val):
                if bmi_val < 18.5: return 0
                elif bmi_val < 25: return 1
                elif bmi_val < 30: return 2
                else: return 3

            def bp_range(bp_val):
                if bp_val < 80: return 0
                elif bp_val < 90: return 1
                else: return 2

            raw_input['BMI_Category'] = bmi_category(bmi)
            raw_input['BP_Range'] = bp_range(blood_pressure)
            raw_input['Glucose_Insulin_Ratio'] = glucose / (insulin + 1)

            # Create DataFrame with correct column order
            input_df = pd.DataFrame([raw_input])[feature_cols]

            # Scale and predict
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]

            # Display results
            st.markdown("### Results")
            if prediction == 1:
                st.error(f"⚠️ **Higher Risk of Diabetes** — Probability: {probability[1]*100:.1f}%")
            else:
                st.success(f"✅ **Lower Risk of Diabetes** — Probability: {probability[0]*100:.1f}%")

            # Confidence bar
            st.markdown("**Confidence breakdown:**")
            col_a, col_b = st.columns(2)
            col_a.metric("Low Risk", f"{probability[0]*100:.1f}%")
            col_b.metric("High Risk", f"{probability[1]*100:.1f}%")

            st.caption("⚠️ This tool is for educational purposes only and is not a substitute for professional medical advice.")

# ─── Model Performance Page ──────────────────────────────────
elif page == "Model Performance":
    st.title("📊 Model Performance")

    try:
        _, _, _, metrics = load_model()

        st.markdown(f"**Best model:** {metrics['best_model']}")
        st.markdown("---")

        st.subheader("Validation Set Comparison")
        comparison = pd.DataFrame({
            'Logistic Regression': metrics['logistic_regression'],
            'Random Forest': metrics['random_forest'],
        }).T
        comparison.columns = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        st.dataframe(comparison.style.highlight_max(axis=0), use_container_width=True)

        st.subheader("Test Set Results (Best Model)")
        test_df = pd.DataFrame([metrics['test_set']])
        test_df.columns = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        st.dataframe(test_df, use_container_width=True)

    except Exception:
        st.warning("Model metrics not available. Run training first.")

# ─── About Page ──────────────────────────────────────────────
elif page == "About":
    st.title("ℹ️ About This Project")
    st.markdown("""
    ### Diabetes Risk Prediction Tool
    **CS 6440 — Health Informatics Practicum | Georgia Tech**

    This application uses machine learning to predict the likelihood of diabetes
    based on diagnostic health measurements from the Pima Indians Diabetes Dataset.

    #### Dataset
    The Pima Indians Diabetes Dataset contains 768 patient records with 8 clinical
    features: Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI,
    Diabetes Pedigree Function, and Age.

    #### Models
    - **Logistic Regression** — interpretable baseline model
    - **Random Forest Classifier** — ensemble model for improved accuracy

    #### Team
    - Vedha Somashekar
    - Abdullah Aljamal
    - Venkat Retineni
    - Anish Lukkireddy
    - Ajay Palankar

    #### Disclaimer
    This tool is developed for academic purposes as part of a health informatics
    course project. It is **not** intended for clinical diagnosis or medical
    decision-making.
    """)
