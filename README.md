# Diabetes Risk Prediction App

**CS 6440 — Health Informatics Practicum**

A machine learning-powered web application that predicts diabetes risk based on patient health metrics, built using the Pima Indians Diabetes Dataset.

## Team
- Vedha Somashekar
- Abdullah Aljamal
- Venkat Retineni
- Anish Lukkireddy
- Ajay Palankar

## Project Structure
```
diabetes-prediction/
├── app/
│   └── streamlit_app.py        # Streamlit web application
├── data/
│   └── preprocess.py           # Data cleaning & feature engineering
├── models/
│   └── train.py                # Model training & evaluation
├── notebooks/                  # Jupyter notebooks for EDA
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download the dataset
Download the [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) and place `diabetes.csv` in the `data/` folder.

### 3. Preprocess data
```bash
python data/preprocess.py
```

### 4. Train models
```bash
python models/train.py
```

### 5. Run the app
```bash
streamlit run app/streamlit_app.py
```

## Deployment
This app is deployed on Streamlit Community Cloud:
**[Live App URL]** *(update after deployment)*

## Models
- Logistic Regression (baseline)
- Random Forest Classifier

Evaluation metrics: Accuracy, Precision, Recall, F1 Score

## Dataset
Pima Indians Diabetes Dataset (768 records, 8 features + target)
- Source: [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
