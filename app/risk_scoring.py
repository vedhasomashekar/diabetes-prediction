"""
risk_scoring.py
---------------
Converts model output probabilities into interpretable risk categories
for the diabetes prediction application.
"""

# Thresholds for risk tiers
LOW_THRESHOLD = 0.30
HIGH_THRESHOLD = 0.60


def get_risk_category(probability: float) -> dict:
    if not (0.0 <= probability <= 1.0):
        raise ValueError(f"Probability must be between 0 and 1, got {probability}")

    probability_pct = round(probability * 100, 1)

    if probability < LOW_THRESHOLD:
        return {
            "category": "Low",
            "label": "🟢 Low Risk",
            "color": "#2ecc71",
            "message": (
                f"Your predicted risk of diabetes is low ({probability_pct}%). "
                "Continue maintaining a healthy lifestyle. "
                "Routine check-ups are still recommended."
            ),
            "probability_pct": probability_pct,
        }
    elif probability < HIGH_THRESHOLD:
        return {
            "category": "Medium",
            "label": "🟡 Medium Risk",
            "color": "#f39c12",
            "message": (
                f"Your predicted risk of diabetes is moderate ({probability_pct}%). "
                "Consider consulting a healthcare provider for further evaluation. "
                "Lifestyle adjustments such as diet and exercise may help reduce risk."
            ),
            "probability_pct": probability_pct,
        }
    else:
        return {
            "category": "High",
            "label": "🔴 High Risk",
            "color": "#e74c3c",
            "message": (
                f"Your predicted risk of diabetes is high ({probability_pct}%). "
                "We strongly recommend consulting a healthcare provider promptly "
                "for a clinical assessment and personalized guidance."
            ),
            "probability_pct": probability_pct,
        }


def render_risk_result(st, probability: float):
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