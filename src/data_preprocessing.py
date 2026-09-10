"""
Data Preprocessing Pipeline for Diabetes Prediction
Based on the methodology from:
"Diabetes Prediction Using Ensembling of Different Machine Learning Classifiers" (IEEE Access, 2020)
and the AML Project Abstract.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

try:
    from config import (
        FEATURE_NAMES,
        TARGET_NAME,
        ZERO_INVALID_COLS,
        RAW_DATASET_PATH,
        MODELS_DIR,
        TEST_SIZE,
        RANDOM_STATE
    )
except ImportError:
    try:
        from src.config import (
            FEATURE_NAMES,
            TARGET_NAME,
            ZERO_INVALID_COLS,
            RAW_DATASET_PATH,
            MODELS_DIR,
            TEST_SIZE,
            RANDOM_STATE
        )
    except ImportError:
        FEATURE_NAMES = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
        TARGET_NAME = 'Outcome'
        ZERO_INVALID_COLS = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        RAW_DATASET_PATH = "diabetes.csv"
        MODELS_DIR = "models"
        TEST_SIZE = 0.2
        RANDOM_STATE = 42


class DiabetesDataPipeline:
    def __init__(self, data_path=None, test_size=TEST_SIZE, random_state=RANDOM_STATE):
        self.data_path = str(data_path or RAW_DATASET_PATH)
        self.test_size = test_size
        self.random_state = random_state
        self.imputer = None
        self.scaler = None
        self.iqr_bounds = {}

    def load_raw_data(self):
        """Loads raw dataset from CSV."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset file not found at: {self.data_path}")
        df = pd.read_csv(self.data_path)
        return df

    def handle_biological_zeros(self, df):
        """Replaces physiological zeros with NaN for appropriate imputation."""
        df_clean = df.copy()
        for col in ZERO_INVALID_COLS:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].replace(0, np.nan)
        return df_clean

    def fit_transform_train(self, X_train):
        """
        Fits imputer, computes IQR boundaries, clips outliers, and fits StandardScaler
        on training data only (to prevent data leakage).
        """
        X_train_clean = X_train.copy()

        # 1. Median Imputation for missing/zero values
        self.imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(
            self.imputer.fit_transform(X_train_clean),
            columns=FEATURE_NAMES,
            index=X_train.index
        )

        # 2. Outlier treatment via IQR rule (Paper Eq. 2)
        # Clipping values to [Q1 - 1.5*IQR, Q3 + 1.5*IQR] to preserve records
        for col in FEATURE_NAMES:
            q1 = X_imputed[col].quantile(0.25)
            q3 = X_imputed[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            self.iqr_bounds[col] = (lower_bound, upper_bound)
            X_imputed[col] = X_imputed[col].clip(lower=lower_bound, upper=upper_bound)

        # 3. Standardization (Z-score normalization, Paper Eq. 4)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_imputed)

        return pd.DataFrame(X_scaled, columns=FEATURE_NAMES, index=X_train.index)

    def transform_test(self, X_test):
        """
        Transforms testing or new inference data using fitted parameters.
        """
        if self.imputer is None or self.scaler is None:
            raise RuntimeError("Pipeline has not been fitted on training data yet.")

        X_test_clean = X_test.copy()
        X_imputed = pd.DataFrame(
            self.imputer.transform(X_test_clean),
            columns=FEATURE_NAMES,
            index=X_test.index
        )

        # Apply same IQR clipping
        for col in FEATURE_NAMES:
            if col in self.iqr_bounds:
                lb, ub = self.iqr_bounds[col]
                X_imputed[col] = X_imputed[col].clip(lower=lb, upper=ub)

        X_scaled = self.scaler.transform(X_imputed)
        return pd.DataFrame(X_scaled, columns=FEATURE_NAMES, index=X_test.index)

    def run_pipeline(self, save_artifacts=True, artifacts_dir=None):
        """
        Full end-to-end preprocessing execution:
        Loads raw data -> Splits train/test -> Fits transformations -> Returns cleaned sets.
        """
        artifacts_dir = str(artifacts_dir or MODELS_DIR)
        raw_df = self.load_raw_data()
        df_zeros_handled = self.handle_biological_zeros(raw_df)

        X = df_zeros_handled[FEATURE_NAMES]
        y = raw_df[TARGET_NAME]

        # Stratified split to maintain class ratio (approx 65% non-diabetic, 35% diabetic)
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )

        X_train_scaled = self.fit_transform_train(X_train_raw)
        X_test_scaled = self.transform_test(X_test_raw)

        if save_artifacts:
            os.makedirs(artifacts_dir, exist_ok=True)
            joblib.dump(self.imputer, os.path.join(artifacts_dir, 'imputer.joblib'))
            joblib.dump(self.scaler, os.path.join(artifacts_dir, 'scaler.joblib'))
            joblib.dump(self.iqr_bounds, os.path.join(artifacts_dir, 'iqr_bounds.joblib'))
            print(f"Preprocessing artifacts successfully saved to '{artifacts_dir}/'")

        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'raw_df': raw_df
        }


def preprocess_single_patient(patient_dict, artifacts_dir=None):
    """
    Utility function to preprocess a single patient's raw features dictionary for real-time inference.
    """
    artifacts_dir = str(artifacts_dir or MODELS_DIR)
    imputer = joblib.load(os.path.join(artifacts_dir, 'imputer.joblib'))
    scaler = joblib.load(os.path.join(artifacts_dir, 'scaler.joblib'))
    iqr_bounds = joblib.load(os.path.join(artifacts_dir, 'iqr_bounds.joblib'))

    df = pd.DataFrame([patient_dict])[FEATURE_NAMES]
    for col in ZERO_INVALID_COLS:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)

    imputed = pd.DataFrame(imputer.transform(df), columns=FEATURE_NAMES)
    for col in FEATURE_NAMES:
        if col in iqr_bounds:
            lb, ub = iqr_bounds[col]
            imputed[col] = imputed[col].clip(lower=lb, upper=ub)

    scaled = scaler.transform(imputed)
    return scaled


if __name__ == "__main__":
    pipeline = DiabetesDataPipeline()
    data = pipeline.run_pipeline()
    print("Preprocessing completed successfully!")
    print(f"X_train shape: {data['X_train'].shape}, X_test shape: {data['X_test'].shape}")
    print(f"y_train class distribution:\n{data['y_train'].value_counts(normalize=True)}")
    print(f"y_test class distribution:\n{data['y_test'].value_counts(normalize=True)}")
