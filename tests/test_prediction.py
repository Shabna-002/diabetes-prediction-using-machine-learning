"""
Unit Tests for Model Prediction and Artifacts
"""

import unittest
import numpy as np
import os
import sys

# Ensure project root and src are on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from predict import load_artifacts, preprocess_patient, predict_patient


class TestModelPrediction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.imputer, cls.scaler, cls.iqr_bounds, cls.models = load_artifacts("models")
        cls.healthy_patient = {
            'Pregnancies': 1,
            'Glucose': 85.0,
            'BloodPressure': 65.0,
            'SkinThickness': 20.0,
            'Insulin': 75.0,
            'BMI': 22.0,
            'DiabetesPedigreeFunction': 0.20,
            'Age': 23.0
        }
        cls.high_risk_patient = {
            'Pregnancies': 6,
            'Glucose': 180.0,
            'BloodPressure': 85.0,
            'SkinThickness': 35.0,
            'Insulin': 200.0,
            'BMI': 38.0,
            'DiabetesPedigreeFunction': 0.90,
            'Age': 55.0
        }

    def test_artifacts_loaded(self):
        """Checks all required serialization artifacts are present and loaded."""
        self.assertIsNotNone(self.imputer)
        self.assertIsNotNone(self.scaler)
        self.assertIsNotNone(self.iqr_bounds)
        self.assertIn('Weighted Soft Voting Ensemble (Proposed)', self.models)
        self.assertIn('Random Forest', self.models)
        self.assertIn('Support Vector Machine', self.models)

    def test_prediction_probabilities(self):
        """Checks predictions return valid probability between 0 and 1."""
        res_healthy = predict_patient(self.healthy_patient)
        res_high_risk = predict_patient(self.high_risk_patient)

        self.assertIn(res_healthy['prediction'], ['Diabetic', 'Non-Diabetic'])
        self.assertIn(res_high_risk['prediction'], ['Diabetic', 'Non-Diabetic'])
        self.assertTrue(0.0 <= res_healthy['probability'] <= 1.0)
        self.assertTrue(0.0 <= res_high_risk['probability'] <= 1.0)

        # Clinically, high risk patient probability should exceed healthy patient probability
        self.assertGreater(res_high_risk['probability'], res_healthy['probability'])


if __name__ == '__main__':
    unittest.main()
