"""
Unit Tests for Data Preprocessing Pipeline
"""

import unittest
import numpy as np
import pandas as pd
import os
import sys

# Ensure project root and src are on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_preprocessing import (
    DiabetesDataPipeline,
    FEATURE_NAMES,
    ZERO_INVALID_COLS,
    preprocess_single_patient
)


class TestDiabetesDataPipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = DiabetesDataPipeline()
        self.sample_patient = {
            'Pregnancies': 2,
            'Glucose': 120.0,
            'BloodPressure': 70.0,
            'SkinThickness': 20.0,
            'Insulin': 80.0,
            'BMI': 25.0,
            'DiabetesPedigreeFunction': 0.35,
            'Age': 32.0
        }

    def test_handle_biological_zeros(self):
        """Checks that biological zero values are converted to NaN."""
        df = pd.DataFrame([{
            'Pregnancies': 0,      # Biological 0 is valid for pregnancies
            'Glucose': 0,          # Invalid 0
            'BloodPressure': 0,    # Invalid 0
            'SkinThickness': 0,    # Invalid 0
            'Insulin': 0,          # Invalid 0
            'BMI': 0.0,            # Invalid 0
            'DiabetesPedigreeFunction': 0.5,
            'Age': 25
        }])

        df_cleaned = self.pipeline.handle_biological_zeros(df)
        self.assertEqual(df_cleaned.loc[0, 'Pregnancies'], 0)
        for col in ZERO_INVALID_COLS:
            self.assertTrue(np.isnan(df_cleaned.loc[0, col]))

    def test_preprocessing_pipeline_execution(self):
        """Checks that pipeline splits, fits, and transforms without nulls."""
        data = self.pipeline.run_pipeline(save_artifacts=False)
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']

        self.assertEqual(X_train.shape[1], len(FEATURE_NAMES))
        self.assertEqual(X_test.shape[1], len(FEATURE_NAMES))
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_test), len(y_test))

        # Check for NaN values in scaled output
        self.assertEqual(X_train.isna().sum().sum(), 0)
        self.assertEqual(X_test.isna().sum().sum(), 0)

    def test_single_patient_preprocessing(self):
        """Checks single patient dictionary preprocessing returns 1x8 normalized array."""
        processed = preprocess_single_patient(self.sample_patient)
        self.assertEqual(processed.shape, (1, 8))
        self.assertFalse(np.isnan(processed).any())


if __name__ == '__main__':
    unittest.main()
