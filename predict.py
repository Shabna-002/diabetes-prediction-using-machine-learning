"""
Diabetes Prediction CLI Tool
Quick inference script for individual patient assessment using trained ML models.
"""

import os
import sys
import argparse
import joblib
import pandas as pd
import numpy as np

# Ensure src directory is accessible
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from data_preprocessing import FEATURE_NAMES, ZERO_INVALID_COLS


def load_artifacts(models_dir="models"):
    """Loads scaler, imputer, iqr bounds and trained models."""
    imputer = joblib.load(os.path.join(models_dir, 'imputer.joblib'))
    scaler = joblib.load(os.path.join(models_dir, 'scaler.joblib'))
    iqr_bounds = joblib.load(os.path.join(models_dir, 'iqr_bounds.joblib'))
    models = joblib.load(os.path.join(models_dir, 'trained_models.joblib'))
    return imputer, scaler, iqr_bounds, models


def preprocess_patient(data_dict, imputer, scaler, iqr_bounds):
    """Preprocesses a single patient dictionary."""
    df = pd.DataFrame([data_dict])[FEATURE_NAMES]
    for col in ZERO_INVALID_COLS:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
    
    imputed = pd.DataFrame(imputer.transform(df), columns=FEATURE_NAMES)
    for col in FEATURE_NAMES:
        if col in iqr_bounds:
            lb, ub = iqr_bounds[col]
            imputed[col] = imputed[col].clip(lower=lb, upper=ub)

    scaled = scaler.transform(imputed)
    return pd.DataFrame(scaled, columns=FEATURE_NAMES)


def predict_patient(patient_dict, model_name="Weighted Soft Voting Ensemble (Proposed)", models_dir="models"):
    """Runs full prediction for a single patient dictionary."""
    imputer, scaler, iqr_bounds, models = load_artifacts(models_dir)

    if model_name not in models:
        available = list(models.keys())
        raise ValueError(f"Model '{model_name}' not found. Available models: {available}")

    model = models[model_name]
    X_scaled = preprocess_patient(patient_dict, imputer, scaler, iqr_bounds)

    prediction = int(model.predict(X_scaled)[0])
    proba = float(model.predict_proba(X_scaled)[0, 1])

    if proba < 0.35:
        risk_level = "Low Risk"
    elif proba < 0.65:
        risk_level = "Moderate Risk"
    else:
        risk_level = "High Risk"

    return {
        'prediction': 'Diabetic' if prediction == 1 else 'Non-Diabetic',
        'prediction_code': prediction,
        'probability': round(proba, 4),
        'probability_percent': f"{proba * 100:.1f}%",
        'risk_level': risk_level,
        'model_used': model_name
    }


def main(args_list=None):
    parser = argparse.ArgumentParser(description="Predict diabetes risk for a patient.", allow_abbrev=False)
    parser.add_argument('--pregnancies', type=float, default=None, help='Number of pregnancies')
    parser.add_argument('--glucose', type=float, default=None, help='Plasma glucose concentration (mg/dL)')
    parser.add_argument('--blood_pressure', type=float, default=None, help='Diastolic blood pressure (mm Hg)')
    parser.add_argument('--skin_thickness', type=float, default=None, help='Triceps skin fold thickness (mm)')
    parser.add_argument('--insulin', type=float, default=None, help='2-Hour serum insulin (mu U/ml)')
    parser.add_argument('--bmi', type=float, default=None, help='Body mass index (weight in kg/(height in m)^2)')
    parser.add_argument('--pedigree', type=float, default=None, help='Diabetes pedigree function')
    parser.add_argument('--age', type=float, default=None, help='Age (years)')
    parser.add_argument('--model', type=str, default="Weighted Soft Voting Ensemble (Proposed)", help='Model name')

    if args_list is not None:
        args = parser.parse_args(args_list)
    else:
        args = parser.parse_args()

    # If no CLI args provided, run demonstration test cases
    if args.glucose is None:
        print("="*65)
        print("   DIABETES PREDICTION SYSTEM - DEMONSTRATION TEST CASES")
        print("="*65)

        test_cases = [
            {
                'name': 'Case 1: Low-Risk Healthy Profile',
                'data': {
                    'Pregnancies': 1,
                    'Glucose': 88.0,
                    'BloodPressure': 66.0,
                    'SkinThickness': 23.0,
                    'Insulin': 94.0,
                    'BMI': 22.5,
                    'DiabetesPedigreeFunction': 0.23,
                    'Age': 24.0
                }
            },
            {
                'name': 'Case 2: High-Risk Profile (Elevated Glucose & BMI)',
                'data': {
                    'Pregnancies': 6,
                    'Glucose': 175.0,
                    'BloodPressure': 84.0,
                    'SkinThickness': 35.0,
                    'Insulin': 210.0,
                    'BMI': 36.8,
                    'DiabetesPedigreeFunction': 0.85,
                    'Age': 52.0
                }
            }
        ]

        for case in test_cases:
            print(f"\nEvaluating: {case['name']}")
            print("Patient Metrics:", case['data'])
            result = predict_patient(case['data'], model_name=args.model)
            print(f"-> Diagnostic Outcome: {result['prediction']}")
            print(f"-> Risk Probability:  {result['probability_percent']} ({result['risk_level']})")
            print(f"-> Model Applied:      {result['model_used']}")
        print("\n" + "="*65)
        print("To predict custom patient values via CLI, use:")
        print("  py -3.10 predict.py --glucose 150 --bmi 33.2 --age 48 --pregnancies 3")
        print("="*65)
        return

    patient_dict = {
        'Pregnancies': args.pregnancies if args.pregnancies is not None else 0.0,
        'Glucose': args.glucose,
        'BloodPressure': args.blood_pressure if args.blood_pressure is not None else 72.0,
        'SkinThickness': args.skin_thickness if args.skin_thickness is not None else 23.0,
        'Insulin': args.insulin if args.insulin is not None else 100.0,
        'BMI': args.bmi if args.bmi is not None else 32.0,
        'DiabetesPedigreeFunction': args.pedigree if args.pedigree is not None else 0.47,
        'Age': args.age if args.age is not None else 33.0
    }

    result = predict_patient(patient_dict, model_name=args.model)
    print("\n--- CLINICAL PREDICTION RESULT ---")
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
