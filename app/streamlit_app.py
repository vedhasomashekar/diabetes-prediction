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

from risk_scoring import render_risk_result

# ─── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
)

# ─── Custom CSS ──────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #eef6ff 0%, #f8fbff 100%);
    border-right: 1px solid #dbeafe;
}

.sidebar-subtitle {
    color: #475569;
    font-size: 0.95rem;
    margin-bottom: 18px;
}

.sidebar-footer {
    margin-top: 35px;
    padding: 14px;
    background: #ffffff;
    border-radius: 14px;
    border: 1px solid #e5e7eb;
    color: #475569;
    font-size: 0.9rem;
}

.section-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 20px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.04);
}
</style>
""", unsafe_allow_html=True)

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
with st.sidebar:
    st.title("🩺 Diabetes App")
    st.markdown(
        '<div class="sidebar-subtitle">Clinical risk prediction dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["Predict", "Model Performance", "About"],
        captions=[
            "Enter patient metrics",
            "View model results",
            "Project information"
        ]
    )

    st.markdown("---")

    st.markdown(
        """
        <div class="sidebar-footer">
            <strong>CS 6440 Practicum</strong><br>
            Georgia Tech<br>
            ML + Health Informatics
        </div>
        """,
        unsafe_allow_html=True
    )

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

    st.subheader("Patient Input Form")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### 👤 Demographics & Risk Factors")
    col1, col2, col3 = st.columns(3)

    with col1:
        pregnancies = st.number_input(
            "Pregnancies",
            min_value=0,
            max_value=20,
            value=1,
            help="Number of times pregnant"
        )

    with col2:
        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=30,
            help="Age in years"
        )

    with col3:
        dpf = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=2.5,
            value=0.5,
            step=0.01,
            help="Diabetes pedigree function (genetic risk score)"
        )

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### 🧪 Lab Values")
    col4, col5 = st.columns(2)

    with col4:
        glucose = st.number_input(
            "Glucose (mg/dL)",
            min_value=0,
            max_value=300,
            value=120,
            help="Plasma glucose concentration (2hr oral glucose tolerance test)"
        )

    with col5:
        insulin = st.number_input(
            "Insulin (mu U/ml)",
            min_value=0,
            max_value=900,
            value=80,
            help="2-Hour serum insulin"
        )

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### 🫀 Vitals & Body Measurements")
    col6, col7, col8 = st.columns(3)

    with col6:
        blood_pressure = st.number_input(
            "Blood Pressure (mm Hg)",
            min_value=0,
            max_value=200,
            value=70,
            help="Diastolic blood pressure"
        )

    with col7:
        skin_thickness = st.number_input(
            "Skin Thickness (mm)",
            min_value=0,
            max_value=100,
            value=20,
            help="Triceps skin fold thickness"
        )

    with col8:
        bmi = st.number_input(
            "BMI",
            min_value=0.0,
            max_value=70.0,
            value=25.0,
            step=0.1,
            help="Body mass index (weight in kg / height in m^2)"
        )

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    if st.button("🔍 Predict Risk", type="primary", use_container_width=True):
        if model_loaded:
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

            input_df = pd.DataFrame([raw_input])[feature_cols]

            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]

            st.markdown("### Results")

            # Team's updated risk scoring component — keep this unchanged
            render_risk_result(st, probability[1], raw_input)
            
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
    st.markdown("""
    <style>
        .about-card {
            background-color: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 18px;
            padding: 28px;
            margin-bottom: 20px;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.06);
        }

        .about-hero {
            background: linear-gradient(135deg, #eff6ff, #f8fafc);
            border: 1px solid #dbeafe;
            border-radius: 20px;
            padding: 32px;
            margin-bottom: 22px;
        }

        .about-title {
            font-size: 2.2rem;
            font-weight: 800;
            color: #111827;
            margin-bottom: 8px;
        }

        .about-subtitle {
            font-size: 1.05rem;
            color: #4b5563;
            margin-bottom: 0;
        }

        .section-title {
            font-size: 1.25rem;
            font-weight: 700;
            color: #111827;
            margin-bottom: 10px;
        }

        .body-text {
            color: #374151;
            font-size: 1rem;
            line-height: 1.8;
        }

        .chip {
            display: inline-block;
            background: #f3f4f6;
            color: #111827;
            padding: 8px 14px;
            margin: 6px 8px 0 0;
            border-radius: 999px;
            font-size: 0.95rem;
            border: 1px solid #e5e7eb;
        }

        .model-box {
            background: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 16px;
            margin-top: 10px;
        }

        .small-note {
            color: #6b7280;
            font-size: 0.92rem;
            margin-top: 8px;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="about-hero">
        <div class="about-title">ℹ️ About This Project</div>
        <p class="about-subtitle">
            Diabetes risk prediction using machine learning for a health informatics course project.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="about-card">
        <div class="section-title">🩺 Diabetes Risk Prediction Tool</div>
        <p class="body-text"><strong>CS 6440 — Health Informatics Practicum | Georgia Tech</strong></p>
        <p class="body-text">
            This application uses machine learning to predict the likelihood of diabetes
            based on diagnostic health measurements from the Pima Indians Diabetes Dataset.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="about-card">
        <div class="section-title">🧑‍💻 UI & Clinical Output Design Contribution</div>
        <p class="body-text">
            The Streamlit interface was improved with a cleaner sidebar, clearer navigation,
            and a more organized clinical input layout. Patient inputs are grouped into
            demographics, lab values, and vitals/body measurements while keeping the backend
            model inputs unchanged.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="about-card">
        <div class="section-title">📚 Dataset</div>
        <p class="body-text">
            The Pima Indians Diabetes Dataset contains 768 patient records with 8 clinical
            features: Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI,
            Diabetes Pedigree Function, and Age.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="about-card">
        <div class="section-title">🤖 Models</div>
        <div class="model-box">
            <strong>Logistic Regression</strong><br>
            <span class="small-note">Interpretable baseline model</span>
        </div>
        <div class="model-box">
            <strong>Random Forest Classifier</strong><br>
            <span class="small-note">Ensemble model for improved accuracy</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="about-card">
        <div class="section-title">👥 Team</div>
        <span class="chip">Vedha Somashekar</span>
        <span class="chip">Abdullah Aljamal</span>
        <span class="chip">Venkat Retineni</span>
        <span class="chip">Anish Lukkireddy</span>
        <span class="chip">Ajay Palankar</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="about-card">
        <div class="section-title">⚠️ Disclaimer</div>
        <p class="body-text">
            This tool is developed for academic purposes as part of a health informatics
            course project. It is <strong>not</strong> intended for clinical diagnosis or
            medical decision-making.
        </p>
    </div>
    """, unsafe_allow_html=True)