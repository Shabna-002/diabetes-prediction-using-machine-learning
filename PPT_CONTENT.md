# PPT — Diabetes Prediction Using Machine Learning
## Slide 1 — Title
- Diabetes Prediction Using Machine Learning
- Machine Learning • Healthcare Analytics • Flask Web Application
- Student Name / Register Number / Department
- College / University / Guide Name

## Slide 2 — Abstract & Problem Statement
- Diabetes is a chronic health condition with severe long-term complications if undetected.
- Objective: Predict individual diabetes risk using clinical and demographic measurements.
- Challenge: Incomplete clinical data, non-linear relationships, and need for transparent decision support.

## Slide 3 — Project Objectives
- Impute physiologically missing clinical values (zeros in glucose, BP, insulin, BMI).
- Standardize clinical features with StandardScaler.
- Train and evaluate 4 Machine Learning classifiers:
  1. Logistic Regression
  2. Decision Tree
  3. Random Forest
  4. Support Vector Machine (SVM)
- Benchmark using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
- Deploy interactive Flask UI with dynamic model selection and database logging.

## Slide 4 — Clinical Dataset (Pima Indians Format)
- Total Samples: 768 patient records
- 8 Input Variables:
  - Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age
- Binary Target: Outcome (0 = Non-diabetic, 1 = Diabetic)
- Class Distribution: 500 Non-diabetic (65.1%), 268 Diabetic (34.9%)

## Slide 5 — Data Cleaning & Preprocessing
- Biologically implausible 0 values identified in Glucose, BP, Skin Thickness, Insulin, and BMI.
- Median Imputation applied to handle missing values robustly against outliers.
- Feature Scaling: StandardScaler ($z = (x - \mu)/\sigma$) ensures distance and gradient stability.
- Stratified 80/20 train-test split (614 train / 154 test).

## Slide 6 — Evaluated Machine Learning Algorithms
1. **Logistic Regression**: Linear probability boundary using sigmoid activation.
2. **Decision Tree**: Non-parametric greedy recursive partitioning based on Gini impurity.
3. **Random Forest**: Ensemble of 100 decorrelated decision trees using bagging and subspace sampling.
4. **Support Vector Machine (SVM)**: Maximum-margin separation with Radial Basis Function (RBF) kernel.

## Slide 7 — Performance Benchmark Results (Test Set N=154)
| Algorithm | Accuracy | Precision | Recall | F1-Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree** | **76.62%** | **65.00%** | **72.22%** | **0.6842** | ★ **Top Model** |
| **Support Vector Machine (SVM)** | 74.03% | 65.22% | 55.56% | 0.6000 | Evaluated |
| **Random Forest** | 74.03% | 66.67% | 51.85% | 0.5833 | Evaluated |
| **Logistic Regression** | 70.78% | 60.00% | 50.00% | 0.5455 | Evaluated |

## Slide 8 — Confusion Matrix Analysis
- **Decision Tree**: TN=79, FP=21, FN=15, TP=39 (Highest Recall: 72.22% - best at catching diabetic risk cases).
- **SVM**: TN=84, FP=16, FN=24, TP=30
- **Random Forest**: TN=86, FP=14, FN=26, TP=28 (Highest True Negative rate: 86%).
- **Logistic Regression**: TN=82, FP=18, FN=27, TP=27

## Slide 9 — System Architecture
- User Browser → Responsive Web UI
- Flask Backend (`app.py`) → Routing, Form Validation & Model Dispatcher
- Pipeline Execution → Median Imputer + StandardScaler + Pickled Model Pipeline
- Database Persistence → MySQL (Primary) with automated SQLite fallback (`predictions.db`)

## Slide 10 — Web Application Features
- **Home**: Project introduction and key statistics.
- **Algorithms Benchmark (`/models`)**: Real-time diagnostic cards, comparison table, and confusion matrices.
- **Prediction Form (`/predict`)**: Input validation + algorithm selector dropdown.
- **Prediction Result**: Risk category, class probability meter, and feature audit.
- **Dashboard & History**: Summary statistics, risk distributions, and log of previous runs.

## Slide 11 — Key Advantages
- Evaluates and benchmarks multiple ML paradigms (linear, tree, ensemble, margin).
- Resilient database layer runs seamlessly anywhere with zero configuration hurdles.
- Transparent reporting with confusion matrices and recall prioritization.

## Slide 12 — Limitations & Ethical Considerations
- Academic screening demonstration, NOT a medical diagnostic instrument.
- Historical dataset demographics may not represent all populations.
- Clinical diagnoses require confirmation by certified healthcare practitioners.

## Slide 13 — Future Scope
- Hyperparameter tuning via Bayesian Optimization or Grid Search.
- Explainable AI (SHAP / LIME) for feature importance visualization.
- User authentication and role-based clinical practitioner accounts.
- Cloud deployment on AWS / GCP / Azure with Docker containers.

## Slide 14 — Conclusion
- Successfully built an end-to-end multi-model diabetes prediction platform.
- Compared 4 machine learning algorithms and established Decision Tree as the top performer for screening sensitivity.
- Successfully deployed via Flask with full auditability and database persistence.

## Slide 15 — Q&A / Thank You
- Questions and Discussion

