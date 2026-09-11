# Diabetes Prediction Using Machine Learning & Ensembling

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.7+-orange.svg)](https://scikit-learn.org/)
[![Tests Passing](https://img.shields.io/badge/tests-10%20passed-brightgreen.svg)]()
[![Live Demo](https://img.shields.io/badge/live%20demo-GitHub%20Pages-success.svg)](https://shabna-002.github.io/diabetes-prediction-using-machine-learning/)
[![Streamlit App](https://img.shields.io/badge/streamlit-1.30+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end clinical decision-support system and Machine Learning research platform for predicting Type 2 Diabetes risk. Integrates 10 machine learning architectures, a ROC-AUC weighted soft voting ensemble, and real-time clinical biomarker stratification (HbA1c, Biological Sex / Gender, Physical Activity, Smoking Status, and Age).

---

## 🌐 Live Access & Deployment Links

| Resource | Access Link | Description |
| :--- | :--- | :--- |
| 🔗 **Live Web Portal** | **[GitHub Pages Deployment](https://shabna-002.github.io/diabetes-prediction-using-machine-learning/)** | Client-side real-time interactive clinical decision portal |
| 📂 **GitHub Repository** | **[GitHub Source Code](https://github.com/Shabna-002/diabetes-prediction-using-machine-learning)** | Complete version-controlled source code repository |
| 💻 **Local Portal** | `http://localhost:5000` | Zero-dependency Python standard library HTTP/REST portal |
| 📊 **Streamlit Dashboard**| `http://localhost:8501` | Multi-tab exploratory analytics and batch prediction suite |

---

## 📌 Standard Architecture & Workflow

```
                   +---------------------------------------+
                   |       Pima Indians Diabetes Data      |
                   |       (768 Records, 8 Primary Feats)  |
                   +-------------------+-------------------+
                                       |
                                       v
                   +---------------------------------------+
                   |        Preprocessing Pipeline         |
                   |  - Biological Zeros -> NaN Imputation |
                   |  - Median Imputer (Fit on Train)      |
                   |  - IQR Outlier Bounding [Q1-1.5, Q3+1.5]|
                   |  - Z-Score Standardization (N(0, 1))  |
                   +-------------------+-------------------+
                                       |
                                       v
                   +---------------------------------------+
                   |       Stratified 80/20 Train-Test     |
                   |       & 5-Fold Cross-Validation       |
                   +-------------------+-------------------+
                                       |
          +----------------------------+---------------------------+
          |                            |                           |
          v                            v                           v
  +---------------+            +---------------+           +---------------+
  | Base Models   |            | Meta-Learner  |           | Proposed Soft |
  | - LogReg      |            | Stacking      |           | Voting Model  |
  | - SVM (RBF)   |            | Classifier    |           | (ROC-AUC      |
  | - DecTree     |            +---------------+           | Weighted)     |
  | - RandForest  |                                        +---------------+
  | - KNN, AdaBst |                                                |
  | - GradBoost   |                                                |
  | - Naive Bayes |                                                |
  +-------+-------+                                                |
          +----------------------------+---------------------------+
                                       |
                                       v
                   +---------------------------------------+
                   |       Evaluation & Benchmarking       |
                   |  - Accuracy, Precision, Recall, F1    |
                   |  - Specificity, ROC-AUC (Held-out)    |
                   |  - Confusion Matrices & ROC Curves    |
                   +-------------------+-------------------+
                                       |
          +----------------------------+---------------------------+
          |                            |                           |
          v                            v                           v
+--------------------+       +--------------------+      +--------------------+
| Responsive Portal  |       | Streamlit App      |      | CLI Inference      |
| index.html         |       | app.py             |      | predict.py         |
| (REST / Offline)   |       | (Port 8501)        |      | (Terminal Engine)  |
+--------------------+       +--------------------+      +--------------------+
```

---

## 🏥 Clinical Diagnostic Features & Lifestyle Biomarkers

### 1. Primary Machine Learning Features (Pima Dataset)
| Feature Name | Clinical Description | Reference Normal Range |
| :--- | :--- | :--- |
| **Pregnancies** | Number of gestational pregnancies | 0 - 17 (Auto-zeroed for Male) |
| **Glucose** | 2-hour plasma glucose concentration (OGTT) | Normal: <100 mg/dL, Pre-diabetic: 100-125, Diabetic: ≥126 |
| **BloodPressure** | Diastolic blood pressure (mm Hg) | Normal: <80 mm Hg, Elevated: ≥80 mm Hg |
| **SkinThickness** | Triceps skinfold thickness (mm) | Subcutaneous adipose tissue indicator |
| **Insulin** | 2-Hour serum insulin level (μU/mL) | Normal fasting: 16 - 166 μU/mL |
| **BMI** | Body Mass Index ($kg/m^2$) | Normal: 18.5 - 24.9, Overweight: 25 - 29.9, Obese: ≥30 |
| **DiabetesPedigreeFunction** | Genetic history risk pedigree score | Continuous genetic score [0.05 - 2.50] |
| **Age** | Chronological patient age (years) | Young: <35, Mid-Adult: 35-44, Guideline Screening: ≥45 |
| **Outcome** | Clinical diagnosis ground truth | 0: Non-Diabetic, 1: Diabetic |

### 2. Clinical Biomarkers & Lifestyle Factors
- **HbA1c Glycated Hemoglobin (%)**: Gold-standard 3-month glycemic marker bi-directionally synchronized via the ADA formula:
  $$\text{eAG (mg/dL)} = 28.7 \times \text{HbA1c} - 46.7$$
- **Patient Biological Sex / Gender**:
  - **Female (Gestational Screening)**: Activates pregnancy history evaluation.
  - **Male (Visceral Profile)**: Enforces `Pregnancies = 0` and evaluates visceral adiposity risk.
- **Physical Activity & Exercise Level**:
  - **Sedentary (`<30 min/wk`)**: Elevated metabolic and insulin resistance risk.
  - **Light (`30-149 min/wk`)**: Sub-optimal activity; below ADA guideline.
  - **Moderate (`150-299 min/wk`)**: Meets WHO/ADA recommended threshold.
  - **Active (`300+ min/wk`)**: Cardiovascular protective with heightened metabolic clearance.
- **Smoking & Tobacco Exposure Status**:
  - **Never Smoked**: Baseline endothelial and metabolic health.
  - **Former Smoker**: Tobacco cessation phase with progressive vascular recovery.
  - **Current Smoker**: 30–40% elevated T2D risk with active arterial inflammation.

---

## 📊 Benchmark Model Performance

Evaluated on held-out stratified test data ($N=154$):

| Classifier Architecture | 5-Fold CV Acc | CV ROC-AUC | Test Accuracy | Test Precision | Test Recall | Test Specificity | Test F1-Score | Test ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **AdaBoost** | 76.39% | 0.827 | **75.32%** | **69.05%** | 53.70% | **87.00%** | **0.6042** | **0.8262** |
| **Gradient Boosting** | 74.59% | 0.812 | 74.68% | 67.44% | 53.70% | 86.00% | 0.5979 | 0.8185 |
| **Weighted Soft Voting Ensemble (Proposed)** | 77.03% | **0.840** | 73.38% | 64.44% | 53.70% | 84.00% | 0.5859 | **0.8170** |
| **Random Forest** | 77.04% | 0.831 | 74.03% | 67.50% | 50.00% | 87.00% | 0.5745 | 0.8169 |
| **Stacking Classifier** | **77.69%** | **0.847** | 72.08% | 62.22% | 51.85% | 83.00% | 0.5657 | 0.8135 |
| **Support Vector Machine (SVM)** | **78.34%** | 0.837 | 73.38% | 64.44% | 53.70% | 84.00% | 0.5859 | 0.8096 |
| **Logistic Regression** | **78.34%** | **0.850** | 71.43% | 60.87% | 51.85% | 82.00% | 0.5600 | 0.8085 |
| **Decision Tree** | 71.82% | 0.753 | **76.62%** | 64.52% | **74.07%** | 78.00% | **0.6897** | 0.7932 |
| **K-Nearest Neighbors** | 75.40% | 0.804 | 75.32% | 66.67% | 59.26% | 84.00% | 0.6275 | 0.7904 |
| **Naive Bayes** | 77.03% | 0.840 | 70.78% | 57.89% | 61.11% | 76.00% | 0.5946 | 0.7935 |

---

## 📁 Standardized Project Structure

```
diabetes-prediction-ml/
├── .gitignore                         # Standard Python, OS & temporary file ignores
├── LICENSE                            # MIT Open Source License
├── pyproject.toml                     # Standard PEP 517/518 build configuration
├── requirements.txt                   # Pinned dependency specifications
├── README.md                          # Standard comprehensive documentation
├── index.html                         # Root web application (GitHub Pages live entrypoint)
├── server.py                          # Dedicated Python standard library HTTP & REST API server
├── app.py                             # Streamlit Clinical Intelligence Dashboard
├── main.py                            # Unified end-to-end training and evaluation pipeline
├── predict.py                         # CLI patient risk inference script
├── diabetes.csv                       # Standardized Pima Indians Diabetes dataset
├── run_app.bat                        # Windows one-click launcher for Streamlit
├── run_website.bat                    # Windows one-click launcher for Web Portal
├── run_pipeline.bat                   # Windows one-click pipeline execution
├── data/
│   ├── raw/
│   │   └── diabetes.csv               # Raw dataset archive
│   └── processed/
│       ├── train_preprocessed.csv     # Preprocessed training dataset
│       └── test_preprocessed.csv      # Preprocessed testing dataset
├── docs/
│   ├── AML_Project_Abstract.pdf       # Academic project abstract
│   ├── IEEE_Reference_Paper.pdf       # Reference research publication
│   ├── PROJECT_REPORT.md              # University-standard academic project report
│   └── VIVA_QUESTIONS_AND_ANSWERS.md  # Comprehensive defense viva Q&A guide
├── models/
│   ├── trained_models.joblib          # Serialized dictionary of all 10 trained models
│   ├── processed_datasets.joblib      # Partitioned train/test arrays
│   ├── imputer.joblib                 # Serialized median imputer
│   ├── scaler.joblib                  # Serialized standard scaler
│   ├── iqr_bounds.joblib              # Serialized IQR boundary values
│   └── model_metrics.json             # Cross-validation and test benchmark metrics
├── notebooks/
│   └── Diabetes_Prediction_Pipeline.ipynb # Jupyter / Google Colab interactive notebook
├── reports/
│   ├── metrics_summary.csv            # Tabular performance matrix
│   └── figures/
│       ├── roc_curves.png             # Multi-model ROC curves
│       ├── confusion_matrices.png     # 3x3 confusion matrix grid
│       ├── model_comparison.png       # Benchmark comparison chart
│       ├── feature_importance.png     # Feature importance rankings
│       └── correlation_heatmap.png    # Clinical feature correlation matrix
├── src/
│   ├── __init__.py                    # Package initializer
│   ├── config.py                      # Global hyperparameters, random seeds, paths
│   ├── data_preprocessing.py          # Data imputation, IQR bounds, scaling pipeline
│   ├── model_training.py              # Stratified 5-fold CV, model fitting, ensembling
│   └── evaluate_and_visualize.py      # Metric computation, ROC & confusion matrix plots
├── tests/
│   ├── __init__.py                    # Test suite initializer
│   ├── test_preprocessing.py          # Unit tests for preprocessing pipeline
│   ├── test_prediction.py             # Unit tests for ML model inference
│   └── test_clinical_evaluation.py    # Unit tests for lifestyle & biomarker evaluations
└── website/
    └── index.html                     # Web portal source template
```

---

## 🚀 Execution & Quickstart Guide

### 1. Environment Setup
Clone the repository and install required packages:
```bash
git clone https://github.com/Shabna-002/diabetes-prediction-using-machine-learning.git
cd diabetes-prediction-using-machine-learning
py -3.10 -m pip install -r requirements.txt
```

### 2. Run Automated Unit Tests (10/10 Passed)
Execute the automated test suite verifying preprocessing, model inference, and clinical evaluations:
```bash
py -3.10 -m unittest discover -s tests -p "test_*.py" -v
```

### 3. Launch Standalone Web Portal (Zero External Dependencies)
```bash
py -3.10 server.py
```
*(Or double-click `run_website.bat`)*  
Access the web portal at **`http://localhost:5000`** in any modern web browser.

### 4. Launch Streamlit Analytics Dashboard
```bash
py -3.10 -m streamlit run app.py
```
*(Or double-click `run_app.bat`)*  
Access the Streamlit dashboard at **`http://localhost:8501`**.

### 5. Command-Line Patient Scoring (CLI)
```bash
# Evaluate sample preset patients:
py -3.10 predict.py

# Evaluate a custom patient record:
py -3.10 predict.py --glucose 145 --bmi 31.5 --age 46 --blood_pressure 80
```

---

## 📡 REST API Reference

The server exposes standard JSON REST endpoints:

### `POST /api/predict`
Calculates real-time patient diabetes probability and clinical biomarker evaluation.

**Request Payload:**
```json
{
  "Gender": "Male",
  "PhysicalActivity": "Moderate",
  "SmokingStatus": "Never",
  "Pregnancies": 0,
  "Glucose": 120,
  "BloodPressure": 70,
  "SkinThickness": 20,
  "Insulin": 80,
  "BMI": 25.0,
  "DiabetesPedigreeFunction": 0.45,
  "Age": 33,
  "HbA1c": 5.8,
  "model_name": "Weighted Soft Voting Ensemble (Proposed)"
}
```

**Response Payload (HTTP 200):**
```json
{
  "prediction": "Non-Diabetic",
  "prediction_code": 0,
  "probability": 0.185,
  "probability_percent": "18.5%",
  "risk_tier": "Low Risk",
  "risk_class": "low",
  "color": "#10B981",
  "recommendation": "Maintain regular physical activity, balanced nutrition, and routine annual health checkups.",
  "model_used": "Weighted Soft Voting Ensemble (Proposed)",
  "biomarkers": [
    { "name": "Biological Sex / Gender", "value": "Male", "status": "Visceral Profile" },
    { "name": "Physical Activity & Exercise", "value": "Moderate (150-299 min/wk)", "status": "Meets ADA Target" },
    { "name": "Smoking & Tobacco Status", "value": "Never Smoked", "status": "Optimal Profile" },
    { "name": "Glycated Hemoglobin (HbA1c)", "value": "5.8%", "status": "Pre-diabetic" },
    { "name": "Fasting Plasma Glucose", "value": "120.0 mg/dL", "status": "Pre-diabetic" }
  ],
  "multi_model_comparison": {
    "Logistic Regression": { "prediction": "Non-Diabetic", "probability_percent": "14.2%" },
    "Decision Tree": { "prediction": "Non-Diabetic", "probability_percent": "20.0%" },
    "Random Forest": { "prediction": "Non-Diabetic", "probability_percent": "18.1%" },
    "Support Vector Machine": { "prediction": "Non-Diabetic", "probability_percent": "19.3%" },
    "Weighted Soft Voting Ensemble (Proposed)": { "prediction": "Non-Diabetic", "probability_percent": "18.5%" }
  }
}
```

### Other Endpoints:
- `GET /api/models`: Returns list of all trained models and their cross-validation accuracies.
- `GET /api/metrics`: Returns full 5-fold cross-validation and test benchmark metrics matrix.
- `POST /api/batch`: Evaluates an array of patient records in a single call.

---

## 📜 Academic References

1. **Hasan, M. K., Alam, M. A., Das, D., Hossain, E., & Hasan, M.** (2020). *Diabetes Prediction Using Ensembling of Different Machine Learning Classifiers*. **IEEE Access**, 8, 76516-76531.
2. **American Diabetes Association (ADA)**. (2024). *Standards of Medical Care in Diabetes—2024*. Diabetes Care, 47(Suppl. 1), S1-S343.
3. **Smith, J. W., Everhart, J. E., Dickson, W. C., Knowler, W. C., & Johannes, R. S.** (1988). *Using the ADAP learning algorithm to forecast the onset of diabetes mellitus*. Proceedings of the Annual Symposium on Computer Application in Medical Care.

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
