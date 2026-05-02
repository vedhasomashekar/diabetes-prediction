# Diabetes Risk Prediction App
**CS 6440 — Health Informatics Practicum | Georgia Tech**

A machine learning-powered web application that predicts diabetes risk based on patient health metrics. The app supports both manual input and live FHIR-based patient data retrieval from a SMART on FHIR R4 server.

## Team
- Vedha Somashekar
- Abdullah Aljamal
- Venkat Retineni 
- Anish Lukkireddy
- Ajay Palankar

## Live App
**[Deployed App URL]** *(add Streamlit Cloud link after deployment)*

## Project Structure

```
diabetes-prediction/
├── app/
│   ├── streamlit_app.py         # Main Streamlit web application
│   ├── fhir_client.py           # FHIR server connection (SMART on FHIR R4)
│   ├── fhir_mapper.py           # Maps FHIR resources to model features
│   ├── model_adapter.py         # Connects FHIR output to ML pipeline
│   ├── risk_scoring.py          # Risk categorization and recommendations
│   └── generate_feature_cols.py # Generates feature_cols.pkl
├── data/
│   └── preprocess.py            # Data cleaning & feature engineering
├── models/
│   ├── train.py                 # Model training & evaluation
│   ├── best_model.pkl           # Trained best model
│   ├── scaler.pkl               # Feature scaler
│   ├── feature_cols.pkl         # Feature column names
│   └── metrics.json             # Model performance metrics
├── notebooks/                   # Jupyter notebooks for EDA
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.gatech.edu/vsomashekar3/diabetes-prediction.git
cd diabetes-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download the [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) and place `diabetes.csv` in the `data/` folder.

### 4. Preprocess data
```bash
python data/preprocess.py
```

### 5. Train models
```bash
python models/train.py
```

### 6. Run the app
```bash
streamlit run app/streamlit_app.py
```

## FHIR Integration
The app connects to the public SMART on FHIR R4 server (`https://r4.smarthealthit.org`). On the Predict page, expand the **Load Patient from FHIR Server** section, enter a patient ID, and click **Load from FHIR** to auto-populate the form with real clinical data.

Example patient IDs from the server:
- `024b1b9b-57bc-4fae-9839-fa0656246b41`
- `068973f7-ee9f-48ef-b4ec-fb6092b14302`
- `45a69bf0-7116-4f9f-a1d9-29700439b4a4`

## Models
- Logistic Regression (baseline)
- Random Forest Classifier

Evaluation metrics: Accuracy, Precision, Recall, F1 Score

## Dataset
Pima Indians Diabetes Dataset (768 records, 8 features + target)
- Source: [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)