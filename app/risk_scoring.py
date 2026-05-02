"""
risk_scoring.py
---------------
Converts model output probabilities into interpretable risk categories
and generates patient-specific clinical recommendations.
"""

LOW_THRESHOLD = 0.30
HIGH_THRESHOLD = 0.60


def get_recommendations(patient_inputs: dict) -> list:
    """
    Generate patient-specific recommendations based on clinical values.
    """
    recs = []

    glucose = patient_inputs.get('Glucose', 0)
    bmi = patient_inputs.get('BMI', 0)
    blood_pressure = patient_inputs.get('BloodPressure', 0)
    age = patient_inputs.get('Age', 0)
    insulin = patient_inputs.get('Insulin', 0)

    if glucose >= 140:
        recs.append("🩸 Your glucose level is elevated. Consider consulting a provider about an A1C test.")
    elif glucose >= 100:
        recs.append("🩸 Your glucose is in the pre-diabetic range. Reducing sugar and refined carb intake may help.")

    if bmi >= 30:
        recs.append("⚖️ Your BMI indicates obesity, a key diabetes risk factor. A structured weight management plan is recommended.")
    elif bmi >= 25:
        recs.append("⚖️ Your BMI is in the overweight range. Regular physical activity and dietary changes can reduce risk.")

    if blood_pressure >= 90:
        recs.append("❤️ Your blood pressure is high. Reducing sodium intake and regular monitoring is advised.")
    elif blood_pressure >= 80:
        recs.append("❤️ Your blood pressure is slightly elevated. Consider lifestyle modifications to bring it down.")

    if age >= 45:
        recs.append("📅 Age is a risk factor for Type 2 diabetes. Annual screening is recommended for adults over 45.")

    if insulin == 0 or insulin < 16:
        recs.append("💉 Insulin data is missing or very low. A fasting insulin test may provide more insight.")

    if not recs:
        recs.append("✅ Your individual indicators look within normal ranges. Keep up your healthy habits!")

    return recs


def get_risk_category(probability: float) -> dict:
    if not (0.0 <= probability <= 1.0):
        raise ValueError(f"Probability must be between 0 and 1, got {probability}")

    probability_pct = round(probability * 100, 1)

    if probability < LOW_THRESHOLD:
        return {
            "category": "Low",
            "label": "🟢 Low Risk",
            "color": "#2ecc71",
            "message": f"Your predicted risk of diabetes is low ({probability_pct}%). Continue maintaining a healthy lifestyle. Routine check-ups are still recommended.",
            "probability_pct": probability_pct,
        }
    elif probability < HIGH_THRESHOLD:
        return {
            "category": "Medium",
            "label": "🟡 Medium Risk",
            "color": "#f39c12",
            "message": f"Your predicted risk of diabetes is moderate ({probability_pct}%). Consider consulting a healthcare provider for further evaluation.",
            "probability_pct": probability_pct,
        }
    else:
        return {
            "category": "High",
            "label": "🔴 High Risk",
            "color": "#e74c3c",
            "message": f"Your predicted risk of diabetes is high ({probability_pct}%). We strongly recommend consulting a healthcare provider promptly for a clinical assessment.",
            "probability_pct": probability_pct,
        }


def render_risk_result(st, probability: float, patient_inputs: dict = None):
    result = get_risk_category(probability)

    st.markdown(
        f"""
        <div style="
            background-color: {result['color']}22;
            border-left: 6px solid {result['color']};
            padding: 1rem 1.5rem;
            border-radius: 8px;
            margin-top: 1rem;
        ">
            <h2 style="color: {result['color']}; margin: 0 0 0.5rem 0;">
                {result['label']}
            </h2>
            <p style="margin: 0; font-size: 1rem;">{result['message']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if patient_inputs:
        recs = get_recommendations(patient_inputs)
        st.markdown("### 📋 Personalized Recommendations")
        for rec in recs:
            st.markdown(f"- {rec}")