# Diabetes Prediction Using Machine Learning & Ensembling

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.7+-orange.svg)](https://scikit-learn.org/)
[![Streamlit App](https://img.shields.io/badge/streamlit-1.63+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end clinical decision-support and Machine Learning system for predicting diabetes risk using diagnostic and demographic parameters from the Pima Indians Diabetes dataset.

---


## 📌 Project Architecture & Workflow

```
                   +---------------------------+
                   | Pima Indians Dataset      |
                   | (768 Patients, 8 Features)|
                   +-------------+-------------+
                                 |
                                 v
                   +---------------------------+
                   | Preprocessing Pipeline    |
                   | - Biological Zeros -> NaN |
                   | - Median Imputation       |
                   | - IQR Outlier Treatment   |
                   | - Z-Score Standardization |
                   +-------------+-------------+
                                 |
                                 v
                   +---------------------------+
                   | Stratified Split (80/20)  |
                   | & 5-Fold Stratified CV    |
                   +-------------+-------------+
                                 |
          +----------------------+-----------------------+
          |                      |                       |
          v                      v                       v
  +---------------+      +---------------+       +---------------+
  | Base Models   |      | Meta-Learner  |       | IEEE Proposed |
  | - LogReg      |      | Stacking      |       | Soft Voting   |
  | - SVM (RBF)   |      | Classifier    |       | Ensemble with |
  | - DecTree     |      +---------------+       | ROC-AUC       |
  | - RandForest  |                              | Weighting     |
  | - KNN, AdaBst |                              +---------------+
  | - GradBoost   |                                      |
  | - Naive Bayes |                                      |
  +-------+-------+                                      |
          +----------------------+-----------------------+
                                 |
                                 v
                   +---------------------------+
                   | Evaluation & Benchmarking |
                   | - Accuracy, Precision     |
                   | - Recall, F1-Score        |
                   | - Specificity, ROC-AUC    |
                   | - Confusion Matrices      |
                   +-------------+-------------+
                                 |
          +----------------------+-----------------------+
          |                                              |
          v                                              v
+--------------------+                        +--------------------+
| Interactive UI     |                        | Command-Line CLI   |
| Streamlit Web App  |                        | predict.py         |
| (app.py)           |                        |                    |
+--------------------+                        +--------------------+
```

---

## 🏥 Clinical Diagnostic Features

| Feature Name | Clinical Description | Reference Range |
| :--- | :--- | :--- |
| **Pregnancies** | Number of times pregnant | 0 - 17 |
| **Glucose** | 2-hour plasma glucose from oral glucose tolerance test | Normal: <100 mg/dL, Pre-diabetic: 100-125, Diabetic: ≥126 |
| **BloodPressure** | Diastolic blood pressure | Normal: <80 mm Hg, Hypertension: ≥80 mm Hg |
| **SkinThickness** | Triceps skin fold thickness | Subcutaneous body fat measure (mm) |
| **Insulin** | 2-Hour serum insulin level | Normal fasting: 15 - 150 μU/mL |
| **BMI** | Body mass index ($kg/m^2$) | Normal: 18.5 - 24.9, Overweight: 25 - 29.9, Obese: ≥30 |
| **DiabetesPedigreeFunction** | Family history genetic risk score | Continuous score [0.05 - 2.5] |
| **Age** | Age in years | 21 - 81 years |
| **Outcome** | Class variable (Ground Truth) | 0: Non-Diabetic, 1: Diabetic |

---

## 🔬 Mathematical Methodology

### 1. Handling Physiological Missing Values
In living individuals, glucose, blood pressure, skin thickness, insulin, and BMI cannot biologically be zero. Zeroes in these fields indicate missing data and are imputed using median values (Paper Eq. 3):
$$Q(x) = \begin{cases} \text{median}(x), & \text{if } x = 0 \text{ or null} \\ x, & \text{otherwise} \end{cases}$$

### 2. Outlier Rejection via Interquartile Range (IQR)
Continuous features are bounded to mitigate the influence of extreme anomalies (Paper Eq. 2):
$$\text{IQR} = Q_3 - Q_1$$
$$\text{Bounds} = [Q_1 - 1.5 \times \text{IQR}, \; Q_3 + 1.5 \times \text{IQR}]$$

### 3. Feature Standardization
Features are transformed to standard normal distribution with zero mean and unit variance ($Z$-score normalization, Paper Eq. 4):
$$Z = \frac{x - \mu}{\sigma}$$

### 4. Weighted Soft Voting Ensembling
Individual base classifiers generate posterior confidence probabilities $P_{ij}$ for each class $i \in \{0, 1\}$. The proposed ensemble aggregates predictions weighted by each classifier's cross-validated ROC-AUC ($W_j$) (Paper Eq. 6):
$$P^{en}_i = \frac{\sum_{j=1}^{m} (W_j \times P_{ij})}{\sum_{i=1}^{C} \sum_{j=1}^{m} (W_j \times P_{ij})}$$

---

## 📊 Benchmark Model Performance

Tested on a held-out stratified test partition ($N=154$):

| Classifier Architecture | CV Accuracy | CV ROC-AUC | Test Accuracy | Test Precision | Test Recall | Test Specificity | Test F1-Score | Test ROC-AUC |
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

## 🚀 Getting Started & Execution Guide

### Prerequisites
Make sure Python 3.10+ is available on your system.

### 1. Install Dependencies
```bash
py -3.10 -m pip install numpy pandas scikit-learn matplotlib seaborn joblib streamlit pypdf
```

### 2. Train Models and Run Benchmark
```bash
py -3.10 src/model_training.py
```
This script runs 5-Fold Stratified Cross-Validation on all models, calculates individual ROC-AUC scores, builds the Weighted Soft Voting Ensemble and Stacking models, and serializes artifacts to `models/`.

### 3. Generate Publication Figures
```bash
py -3.10 src/evaluate_and_visualize.py
```
Outputs saved in `reports/figures/`:
- `roc_curves.png`: Multi-model ROC comparison
- `confusion_matrices.png`: Confusion matrices grid
- `model_comparison.png`: Performance bar chart
- `feature_importance.png`: Clinical driver rankings
- `correlation_heatmap.png`: Feature correlation matrix

### 4A. Standalone Clinical Web Portal (HTML5 / Tailwind / REST API)
Launch the lightweight zero-dependency web application:
```bash
py -3.10 server.py
```
*(Or double-click `run_website.bat`)*
- Open your browser at `http://localhost:5000`.
- **Interactive Risk Gauge**: Real-time diabetes risk probability computed via ML API.
- **Synchronized Sliders & Presets**: Easily test Healthy, Pre-Diabetic, and High-Risk diabetic profiles.
- **Multi-Model Consensus**: Live predictions from SVM, Logistic Regression, Decision Tree, Random Forest, and the Weighted Ensemble.
- **Biomarker Stratification**: Automatic classification of Glucose, BMI, BP, and Insulin vs clinical reference thresholds.
- **Printable Medical Assessment**: Print or save diagnostic summaries as PDF directly from the browser.
- **Batch Screening Demo**: Multi-patient evaluation table.

### 4B. Interactive Streamlit Clinical Dashboard
Launch the Streamlit interface:
```bash
py -3.10 -m streamlit run app.py
```
*(Or double-click `run_app.bat`)*
- Open your browser at `http://localhost:8501`.
- **Tab 1**: Patient Risk Assessment calculator with quick presets.
- **Tab 2**: Batch patient screening from CSV.
- **Tab 3**: Model benchmarks and graphical analytics.
- **Tab 4**: Interactive exploratory data analysis (EDA).
- **Tab 5**: Research paper mathematical formulation.

### 5. Command-Line Inference (CLI)
Test individual patients directly from the terminal:
```bash
# Run demonstration test cases (Healthy vs High-Risk):
py -3.10 predict.py

# Custom patient assessment:
py -3.10 predict.py --glucose 165 --bmi 34.2 --age 49 --pregnancies 4 --blood_pressure 82
```

### 6. Automated Unit Testing
Run the test suite to verify data preprocessing, pipeline transformations, and model predictions:
```bash
py -3.10 -m unittest discover -s tests -p "test_*.py" -v
```

---

## 📁 Standard Repository Structure

```
diabetes-prediction-ml/
├── .gitignore                         # Standard Python gitignore (ignores pycache, build, checkpoints)
├── pyproject.toml                     # Modern standard Python project configuration
├── requirements.txt                   # Pinned Python package dependencies
├── README.md                          # Project overview and execution guide
├── run_app.bat                        # One-click Windows launch for Web UI
├── run_pipeline.bat                   # One-click Windows execution for ML pipeline
├── app.py                             # Interactive Streamlit Web Application
├── main.py                            # Unified master pipeline runner
├── predict.py                         # CLI inference engine for patient scoring
├── diabetes.csv                       # Pima Indians Diabetes dataset
├── data/
│   ├── raw/
│   │   └── diabetes.csv               # Raw dataset copy
│   └── processed/
│       ├── train_preprocessed.csv     # Scaled, imputed training set
│       └── test_preprocessed.csv      # Scaled, imputed testing set
├── docs/
│   ├── AML_Project_Abstract.pdf       # Student academic project abstract
│   ├── IEEE_Reference_Paper.pdf       # IEEE Access 2020 foundation research paper
│   ├── PROJECT_REPORT.md              # University-standard academic project report
│   └── VIVA_QUESTIONS_AND_ANSWERS.md  # Comprehensive Viva Voce / defense Q&A guide
├── models/
│   ├── trained_models.joblib          # Serialized ML model dictionary
│   ├── processed_datasets.joblib      # Partitioned arrays
│   ├── imputer.joblib                 # Fitted median imputer
│   ├── scaler.joblib                  # Fitted StandardScaler
│   ├── iqr_bounds.joblib              # Fitted IQR boundaries
│   └── model_metrics.json             # Cross-validation & test metrics
├── notebooks/
│   └── Diabetes_Prediction_Pipeline.ipynb # Complete academic Jupyter Notebook (Colab ready)
├── reports/
│   ├── metrics_summary.csv            # Official comparative evaluation table
│   └── figures/
│       ├── roc_curves.png             # Multi-model ROC curves
│       ├── confusion_matrices.png     # 3x3 Confusion matrices grid
│       ├── model_comparison.png       # Cross-model performance bar chart
│       ├── feature_importance.png     # Random Forest feature importance
│       └── correlation_heatmap.png    # Clinical feature correlation matrix
├── src/
│   ├── __init__.py                    # Package initialization
│   ├── config.py                      # Centralized configuration & hyperparameter constants
│   ├── data_preprocessing.py          # Biological zero treatment, IQR clipping, scaling
│   ├── model_training.py              # 5-Fold Stratified CV, Model Training & Ensembles
│   └── evaluate_and_visualize.py      # Evaluation metrics, ROC & confusion matrix plots
└── tests/
    ├── __init__.py                    # Test package initialization
    ├── test_preprocessing.py          # Automated unit tests for data cleaning & scaling
    └── test_prediction.py             # Automated unit tests for inference & models
```

---

## 📚 References
1. **Hasan, M. K., Alam, M. A., Das, D., Hossain, E., & Hasan, M.** (2020). *Diabetes Prediction Using Ensembling of Different Machine Learning Classifiers*. IEEE Access, 8, 76516-76531.
2. **Smith, J. W., Everhart, J. E., Dickson, W. C., Knowler, W. C., & Johannes, R. S.** (1988). *Using the ADAP learning algorithm to forecast the onset of diabetes mellitus*. Proceedings of the Annual Symposium on Computer Application in Medical Care.
