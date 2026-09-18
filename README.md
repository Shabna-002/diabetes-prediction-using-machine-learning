# Diabetes Prediction Using Machine Learning

An end-to-end Machine Learning web application evaluating and deploying **Logistic Regression**, **Decision Tree**, **Random Forest**, and **Support Vector Machine (SVM)** for diabetes risk screening.

## Features
- **4 Machine Learning Classifiers**:
  - Logistic Regression
  - Decision Tree Classifier (Best Overall F1-Score)
  - Random Forest Classifier
  - Support Vector Machine (SVM with RBF Kernel)
- **Clinical Data Preprocessing**:
  - Median imputation of biologically implausible zero values (Glucose, Blood Pressure, Skin Thickness, Insulin, BMI)
  - Feature scaling via `StandardScaler`
- **Algorithm Benchmark & Evaluation Dashboard** (`/models`):
  - Accuracy, Precision, Recall, F1-Score comparisons
  - Interactive Confusion Matrices for each algorithm
- **Interactive Flask Web Application**:
  - Live patient risk assessment with model selection dropdown
  - Probability gauges and evaluated clinical feature summary
- **Dual Database Persistence**:
  - MySQL support for enterprise logging
  - Automated SQLite fallback (`data/predictions.db`) for immediate out-of-the-box local operation
- **Complete Academic Documentation**:
  - Comprehensive Project Report in `docs/REPORT.md`
  - Presentation Slide Outline in `docs/PPT_CONTENT.md`
  - 30 Viva Questions and Answers in `docs/VIVA_30.md`

## Important Medical Note
This project is an academic and technical demonstration for educational screening purposes. It is **not** a clinical diagnostic instrument. Final medical diagnoses must always be made by qualified healthcare practitioners.

## Project Structure
```
diabetes_prediction_ml_project/
├── app.py                      # Flask web application & dual-database backend
├── train_model.py              # Multi-model training and benchmarking script
├── requirements.txt            # Python package dependencies
├── database.sql                # MySQL schema definition
├── data/
│   ├── diabetes.csv            # 768-record Pima Indians Diabetes dataset
│   └── predictions.db          # Auto-generated local SQLite fallback database
├── model/
│   ├── diabetes_model.joblib   # Best performing model pipeline
│   ├── decision_tree.joblib    # Trained Decision Tree pipeline
│   ├── random_forest.joblib    # Trained Random Forest pipeline
│   ├── svm.joblib              # Trained SVM pipeline
│   ├── logistic_regression.joblib
│   └── model_metrics.json      # Benchmark evaluation metrics & confusion matrices
├── templates/                  # Jinja2 HTML templates
│   ├── base.html
│   ├── index.html
│   ├── models.html             # Algorithm benchmark & confusion matrix view
│   ├── predict.html            # Prediction form with model selector
│   ├── result.html             # Risk score & probability display
│   ├── dashboard.html          # Aggregate statistics
│   └── history.html            # Prediction logs & database indicator
├── static/                     # CSS stylesheets and client JavaScript
└── docs/                       # Project report, PPT content, and viva questions
```

## Quick Start Guide

### 1. Requirements
Ensure Python 3.10+ is installed on your system.

### 2. Install Dependencies
In PowerShell:
```powershell
py -3.10 -m pip install -r requirements.txt
```

### 3. Train & Benchmark All 4 Models
Run the training script to clean data, train all 4 classifiers, compute evaluation metrics, and export models:
```powershell
py -3.10 train_model.py
```
This benchmarks:
- **Decision Tree**: ~76.62% Accuracy, 0.6842 F1-score (Top Model)
- **Support Vector Machine (SVM)**: ~74.03% Accuracy, 0.6000 F1-score
- **Random Forest**: ~74.03% Accuracy, 0.5833 F1-score
- **Logistic Regression**: ~70.78% Accuracy, 0.5455 F1-score

### 4. Database Setup (Optional)
The application connects to **MySQL** if configured (settings in `app.py` or environment variables `DB_HOST`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`). If MySQL is not running or credentials are not supplied, the app **automatically falls back to SQLite** (`data/predictions.db`) with zero setup required.

### 5. Launch the Web Application
```powershell
py -3.10 app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

## Dataset Specifications
- Source: Pima Indians Diabetes Dataset format (768 patient records).
- 8 Features: Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age.
- Target: Binary Outcome (0 = Non-diabetic, 1 = Diabetic).

