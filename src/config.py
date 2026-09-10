"""
Central Configuration for Diabetes Prediction ML System
Encapsulates directory paths, dataset schemas, clinical boundaries,
hyperparameters, and model serialization constants.
"""

from pathlib import Path

# Project root directory (parent of src)
BASE_DIR = Path(__file__).resolve().parent.parent

# Standard Directory Paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DOCS_DIR = BASE_DIR / "docs"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# File Paths
RAW_DATASET_PATH = BASE_DIR / "diabetes.csv"
TRAIN_PROCESSED_PATH = PROCESSED_DATA_DIR / "train_preprocessed.csv"
TEST_PROCESSED_PATH = PROCESSED_DATA_DIR / "test_preprocessed.csv"

TRAINED_MODELS_PATH = MODELS_DIR / "trained_models.joblib"
IMPUTER_PATH = MODELS_DIR / "imputer.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
IQR_BOUNDS_PATH = MODELS_DIR / "iqr_bounds.joblib"
PROCESSED_DATASETS_PATH = MODELS_DIR / "processed_datasets.joblib"
MODEL_METRICS_PATH = MODELS_DIR / "model_metrics.json"
METRICS_SUMMARY_CSV = REPORTS_DIR / "metrics_summary.csv"

# Clinical Diagnostic Features
FEATURE_NAMES = [
    'Pregnancies',
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI',
    'DiabetesPedigreeFunction',
    'Age'
]

TARGET_NAME = 'Outcome'

# Attributes where zero value is biologically invalid and indicates missing data
ZERO_INVALID_COLS = [
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI'
]

# Clinical Reference Ranges for UI and sanity checks
FEATURE_RANGES = {
    'Pregnancies': {'min': 0, 'max': 17, 'default': 1, 'unit': 'count'},
    'Glucose': {'min': 40.0, 'max': 250.0, 'default': 110.0, 'unit': 'mg/dL'},
    'BloodPressure': {'min': 40.0, 'max': 140.0, 'default': 70.0, 'unit': 'mm Hg'},
    'SkinThickness': {'min': 5.0, 'max': 99.0, 'default': 20.0, 'unit': 'mm'},
    'Insulin': {'min': 10.0, 'max': 800.0, 'default': 80.0, 'unit': 'μU/mL'},
    'BMI': {'min': 15.0, 'max': 65.0, 'default': 25.0, 'unit': 'kg/m²'},
    'DiabetesPedigreeFunction': {'min': 0.05, 'max': 2.5, 'default': 0.35, 'unit': 'score'},
    'Age': {'min': 18, 'max': 100, 'default': 30, 'unit': 'years'}
}

# Reproducibility & Partitioning Settings
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
