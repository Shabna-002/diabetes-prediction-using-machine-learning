# DIABETES PREDICTION USING MACHINE LEARNING

## Abstract
Diabetes is a common chronic disease that can lead to serious health complications if it is not detected and managed at an early stage. This project, “Diabetes Prediction Using Machine Learning,” aims to develop a machine learning-based system for predicting whether a person is likely to have diabetes based on relevant medical and demographic parameters. The system uses a dataset containing features such as pregnancy history, glucose level, blood pressure, skin thickness, insulin level, BMI, diabetes pedigree function, and age. The dataset is preprocessed to handle missing or inconsistent values and is then divided into training and testing sets. Machine learning algorithms such as Logistic Regression, Decision Tree, Random Forest, and Support Vector Machine (SVM) can be trained and evaluated to identify the most suitable model for prediction. Performance is measured using metrics such as accuracy, precision, recall, F1-score, and confusion matrix. The proposed system provides a quick and cost-effective preliminary prediction of diabetes risk. It can serve as a decision-support tool for early screening, while the final diagnosis should always be made by qualified healthcare professionals. The project demonstrates how machine learning can be applied to healthcare data to support early detection and improve preventive healthcare.

## 1. Introduction
Diabetes mellitus is a metabolic disorder characterized by sustained high blood glucose levels. If left undetected, it can cause long-term complications including cardiovascular disease, neuropathy, nephropathy, and retinopathy. Machine learning models can analyze multi-parametric clinical indicators to assist in early detection and risk screening.

This project implements an end-to-end educational machine learning system. It compares four distinct classification algorithms—Logistic Regression, Decision Tree, Random Forest, and Support Vector Machine (SVM)—trained on standardized clinical features, and exposes the models through a responsive Flask web application backed by database storage.

## 2. Problem Statement
Develop a machine learning-based prediction system that:
1. Ingests patient medical and demographic parameters.
2. Cleans and imputes missing or inconsistent clinical data.
3. Evaluates and benchmarks multiple classification algorithms.
4. Provides real-time risk predictions with probability scores.
5. Persists prediction history for analytical auditing and dashboard reporting.

## 3. Objectives
1. **Dataset Preparation**: Structure and analyze the 768-record Pima Indians Diabetes dataset.
2. **Data Cleaning & Preprocessing**: Detect and impute biologically implausible zero values (e.g., Glucose, Blood Pressure, BMI) using median imputation and apply feature standardization (`StandardScaler`).
3. **Multi-Model Training**: Implement and train four machine learning algorithms:
   - Logistic Regression
   - Decision Tree Classifier
   - Random Forest Classifier
   - Support Vector Machine (SVM with RBF kernel)
4. **Comprehensive Evaluation**: Measure and compare test performance using Accuracy, Precision, Recall, F1-score, and Confusion Matrix.
5. **Interactive Web Interface**: Build a Flask web app supporting dynamic model selection and benchmark inspection.
6. **Data Persistence**: Store prediction logs and metrics in a relational database with automatic SQLite fallback for reliable standalone operation.

## 4. Dataset Description
The system is trained on the benchmark Pima Indians Diabetes dataset:
- **Total Instances**: 768 observations
- **Target Variable**: `Outcome` (0 = Non-diabetic, 1 = Diabetic)
- **Input Features (8 parameters)**:
  1. `Pregnancies`: Number of pregnancies
  2. `Glucose`: Plasma glucose concentration at 2 hours in an oral glucose tolerance test (mg/dL)
  3. `BloodPressure`: Diastolic blood pressure (mm Hg)
  4. `SkinThickness`: Triceps skin fold thickness (mm)
  5. `Insulin`: 2-Hour serum insulin (µU/mL)
  6. `BMI`: Body mass index ($weight\ in\ kg / (height\ in\ m)^2$)
  7. `DiabetesPedigreeFunction`: Diabetes pedigree function (genetic score)
  8. `Age`: Patient age in years

## 5. Methodology

### 5.1 Data Preprocessing & Imputation
In clinical datasets, zero values in physiological attributes such as Glucose, Blood Pressure, Skin Thickness, Insulin, and BMI represent missing or unrecorded values rather than true zeros.
- Identified zero values are converted to missing values (`NaN`).
- **Median Imputation** (`SimpleImputer(strategy="median")`) replaces missing entries with robust central-tendency values resistant to outliers.
- **Feature Scaling** (`StandardScaler`) standardizes features by centering the mean to 0 and scaling to unit variance:
  $$z = \frac{x - \mu}{\sigma}$$

### 5.2 Train-Test Split
The cleaned dataset is split into:
- **Training Set (80%)**: 614 samples
- **Testing Set (20%)**: 154 samples
A **stratified split** preserves the class ratio across train and test partitions.

### 5.3 Machine Learning Algorithms

1. **Logistic Regression**:
   Models the log-odds of the diabetic class as a linear combination of input features:
   $$P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \sum \beta_i X_i)}}$$
   Provides high interpretability and probabilistic outputs.

2. **Decision Tree Classifier**:
   Partitions the feature space through recursive binary splits based on Gini impurity:
   $$\text{Gini}(D) = 1 - \sum_{i=1}^{k} p_i^2$$
   Captures nonlinear threshold rules.

3. **Random Forest Classifier**:
   An ensemble of 100 decorrelated decision trees built via bootstrap aggregating (bagging) and random feature subspace selection. Reduces variance and prevents overfitting:
   $$\hat{y} = \text{mode}(\{h_b(x)\}_{b=1}^B)$$

4. **Support Vector Machine (SVM)**:
   Finds the optimal separating hyperplane maximizing the geometric margin between classes. Utilizes a Radial Basis Function (RBF) kernel:
   $$K(x, x') = \exp(-\gamma ||x - x'||^2)$$
   with Platt scaling calibration to produce valid posterior probabilities.

---

## 6. Experimental Results & Performance Comparison

All four models were evaluated on the held-out test set ($N=154$, with 100 Non-diabetic and 54 Diabetic cases).

### 6.1 Performance Benchmark Table

| Machine Learning Model | Accuracy | Precision | Recall | F1-Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree** | **76.62%** | **0.6500** | **0.7222** | **0.6842** | ★ **Top Performer** |
| **Support Vector Machine (SVM)** | 74.03% | 0.6522 | 0.5556 | 0.6000 | Evaluated |
| **Random Forest** | 74.03% | 0.6667 | 0.5185 | 0.5833 | Evaluated |
| **Logistic Regression** | 70.78% | 0.6000 | 0.5000 | 0.5455 | Evaluated |

### 6.2 Confusion Matrix Breakdown

$$\begin{bmatrix} \text{TN} & \text{FP} \\ \text{FN} & \text{TP} \end{bmatrix}$$

1. **Decision Tree**:
   - True Negatives (TN): 79 | False Positives (FP): 21
   - False Negatives (FN): 15 | True Positives (TP): 39
   - *Key finding*: Highest recall (72.22%) and F1-score (0.6842), identifying the highest number of diabetic cases.

2. **Support Vector Machine (SVM)**:
   - TN: 84 | FP: 16
   - FN: 24 | TP: 30

3. **Random Forest**:
   - TN: 86 | FP: 14
   - FN: 26 | TP: 28

4. **Logistic Regression**:
   - TN: 82 | FP: 18
   - FN: 27 | TP: 27

---

## 7. System Architecture & Modules

```
[ User / Browser ]
        │
        ▼
[ Flask Application (app.py) ]
   ├── Routing & Input Validation
   ├── Dynamic Model Dispatcher
   │      ├── Decision Tree (Best Model)
   │      ├── Random Forest
   │      ├── Support Vector Machine (SVM)
   │      └── Logistic Regression
   │
   ├── Pipeline Execution (Imputation + StandardScaler + Inference)
   │
   └── Resilient Database Persistence
          ├── MySQL Server (Primary)
          └── SQLite (Automated Local Fallback)
```

### Application Modules:
- **Home Module (`/`)**: Overview of the ML architecture, features, and model performance.
- **Algorithm Benchmark (`/models`)**: Real-time diagnostic view comparing accuracy, precision, recall, F1-scores, and confusion matrices.
- **Prediction Module (`/predict`)**: Input form allowing clinical parameter entry and classifier selection.
- **Results Module**: Instant risk categorization, predicted class, probability gauge, and input audit summary.
- **History Module (`/history`)**: Audited log of all stored predictions, including the model utilized.
- **Dashboard Module (`/dashboard`)**: Aggregated counts, risk ratio distributions, and database storage status.

---

## 8. Database Design

Table name: `predictions`
- `id` (INTEGER / INT, Primary Key)
- `patient_name` (VARCHAR(100) / TEXT)
- `model_used` (VARCHAR(50) / TEXT)
- `pregnancies` (REAL / INT)
- `glucose` (REAL / FLOAT)
- `blood_pressure` (REAL / FLOAT)
- `skin_thickness` (REAL / FLOAT)
- `insulin` (REAL / FLOAT)
- `bmi` (REAL / FLOAT)
- `diabetes_pedigree` (REAL / FLOAT)
- `age` (REAL / INT)
- `prediction` (INTEGER / TINYINT)
- `probability` (REAL / FLOAT)
- `created_at` (TIMESTAMP DEFAULT CURRENT_TIMESTAMP)

---

## 9. Limitations & Clinical Advisory
This application is developed strictly for educational, research, and technical demonstration purposes. A statistical machine learning classification is not equivalent to a medical diagnosis. Variations in laboratory assays, fasting states, and patient demographic profiles require that actual diagnostic decisions always be confirmed by licensed medical professionals.

---

## 10. Conclusion
The project successfully realizes the objectives outlined in the abstract:
- Evaluated and benchmarked four machine learning algorithms (Logistic Regression, Decision Tree, Random Forest, SVM).
- Successfully handled missing clinical data via median imputation and standardization.
- Implemented an interactive, database-backed web application providing immediate predictions and analytical transparency.

## References
1. Kahn, M. Diabetes [Dataset]. UCI Machine Learning Repository, DOI: 10.24432/C5T59G.
2. Breiman, L. (2001). "Random Forests". *Machine Learning*, 45(1), 5-32.
3. Cortes, C., & Vapnik, V. (1995). "Support-vector networks". *Machine Learning*, 20(3), 273-297.
4. Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
5. Flask Documentation (Pallets Projects).

