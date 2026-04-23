import joblib
import pandas as pd


MODEL = joblib.load("../models/best_model.pkl")
SCALER = joblib.load("../models/scaler.pkl")
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
    "Glucose_Insulin_Ratio"
]

def predict(mapped_features):
    model_input = prepare_model_input(mapped_features)

    pred = MODEL.predict(model_input)[0]
    prob = MODEL.predict_proba(model_input)[0][1]

    return pred, prob

def fill_missing_features(mapped):

    return {
        "Pregnancies": 0,  
        "Glucose": mapped.get("glucose") or 120,
        "BloodPressure": mapped.get("systolic_bp") or 70,
        "SkinThickness": 20, 
        "Insulin": 80,  
        "BMI": mapped.get("bmi") or 30,
        "DiabetesPedigreeFunction": 0.5,  
        "Age": mapped.get("age") or 50,
    }


def engineer_features(row):


    if row["BMI"] < 18.5:
        bmi_cat = 0
    elif row["BMI"] < 25:
        bmi_cat = 1
    elif row["BMI"] < 30:
        bmi_cat = 2
    else:
        bmi_cat = 3


    if row["BloodPressure"] < 80:
        bp_range = 0
    elif row["BloodPressure"] < 90:
        bp_range = 1
    else:
        bp_range = 2


    ratio = row["Glucose"] / (row["Insulin"] + 1)

    row["BMI_Category"] = bmi_cat
    row["BP_Range"] = bp_range
    row["Glucose_Insulin_Ratio"] = ratio

    return row


def prepare_model_input(mapped_features):

    base = fill_missing_features(mapped_features)

    df = pd.DataFrame([base])


    df = df.apply(engineer_features, axis=1)


    df = df[FEATURE_COLS]


    df_scaled = pd.DataFrame(
        SCALER.transform(df),
        columns=FEATURE_COLS
    )

    return df_scaled