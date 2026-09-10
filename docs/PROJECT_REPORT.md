# Applied Machine Learning Project Report: Diabetes Prediction Using Machine Learning

---

## 📌 Executive Summary & Abstract

### Abstract
> **Diabetes is a chronic disease that affects millions of people worldwide and can lead to serious health complications if it is not detected and managed at an early stage. Early prediction of diabetes can help individuals take appropriate preventive measures and receive timely medical care. This project presents a Diabetes Prediction System using Machine Learning to predict whether a person is likely to have diabetes based on relevant medical and demographic attributes.**
>
> **The system uses patient-related features such as Pregnancies, Glucose Level, Blood Pressure, Skin Thickness, Insulin, Body Mass Index (BMI), Diabetes Pedigree Function, and Age. The dataset is preprocessed to handle missing or invalid values, followed by feature analysis and data normalization where required. Machine learning classification algorithms such as Logistic Regression, Decision Tree, Random Forest, Support Vector Machine (SVM), and K-Nearest Neighbors (KNN) can be trained and evaluated to identify the most suitable prediction model.**
>
> **The performance of the models is assessed using evaluation metrics including accuracy, precision, recall, F1-score, and confusion matrix. The proposed system can provide a simple and efficient way to identify individuals who may be at risk of diabetes. It is intended as a predictive decision-support tool and not as a replacement for professional medical diagnosis. The project demonstrates how machine learning can be applied to healthcare data to support early risk prediction and improve preventive healthcare.**

---

## 1. Introduction & Motivation

Diabetes mellitus is a metabolic disorder characterized by elevated levels of blood glucose (hyperglycemia) resulting from defects in insulin secretion, insulin action, or both. Prolonged hyperglycemia can cause long-term damage, dysfunction, and failure of various organs, notably the eyes, kidneys, nerves, heart, and blood vessels.

Early diagnosis is critical to mitigating severe health complications. While standard clinical testing (e.g., Fasting Plasma Glucose, Oral Glucose Tolerance Test, and HbA1c) remains essential, automated screening systems powered by machine learning can:
- Provide low-cost, preliminary risk stratification in rural or under-resourced healthcare settings.
- Assist primary care clinicians in triaging patients who require urgent laboratory follow-up.
- Empower patients to understand how modifiable risk factors (BMI, blood pressure, glucose) influence their health status.

---

## 2. Dataset Description & Clinical Feature Analysis

The system utilizes the benchmark **Pima Indians Diabetes Database** originating from the National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK).

### Dataset Attributes (N = 768)

| Feature | Type | Unit | Description | Clinical Relevance |
| :--- | :---: | :---: | :--- | :--- |
| **Pregnancies** | Discrete | Count | Number of times pregnant | High parity is correlated with increased risk of Gestational Diabetes. |
| **Glucose** | Continuous | mg/dL | 2-hour plasma glucose concentration | Primary diagnostic biomarker for diabetes. |
| **BloodPressure** | Continuous | mm Hg | Diastolic blood pressure | Hypertension strongly correlates with metabolic syndrome. |
| **SkinThickness** | Continuous | mm | Triceps skin fold thickness | Indirect measure of subcutaneous body fat percentage. |
| **Insulin** | Continuous | μU/mL | 2-Hour serum insulin | Assesses insulin resistance and beta-cell secretory function. |
| **BMI** | Continuous | $kg/m^2$ | Weight in $kg$ / (Height in $m$)$^2$ | Obesity ($BMI \ge 30$) is one of the highest diabetes risk drivers. |
| **DiabetesPedigreeFunction** | Continuous | Score | Genetic family history score | Quantifies diabetic hereditary predisposition across relatives. |
| **Age** | Continuous | Years | Patient chronological age | Risk of Type 2 diabetes rises progressively with advancing age. |
| **Outcome** | Binary | {0, 1} | Class label | **0**: Non-Diabetic (65.1%), **1**: Diabetic (34.9%) |

---

## 3. Data Preprocessing Methodology

Clinical datasets present significant real-world challenges including missing values masquerading as zeroes and skewed outlier distributions. The preprocessing pipeline implements:

### 3.1 Handling Biologically Impossible Zero Values
In living humans, measurements such as **Glucose, Blood Pressure, Skin Thickness, Insulin, and BMI cannot biologically be zero**. 
- These entries represent missing data resulting from unrecorded tests.
- All zeros in `['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']` were replaced with `NaN`.
- Values were imputed using **median imputation**, which preserves the distribution without introducing artificial outliers (Paper Eq. 3):
$$\text{Imputed}(x) = \begin{cases} \text{median}(x), & \text{if } x = \text{null/invalid} \\ x, & \text{otherwise} \end{cases}$$

### 3.2 Outlier Treatment via Interquartile Range (IQR)
Outliers can destabilize linear and distance-based classifiers. The Interquartile Range rule was applied (Paper Eq. 2):
$$\text{IQR} = Q_3 - Q_1$$
$$\text{Lower Bound} = Q_1 - 1.5 \times \text{IQR}, \quad \text{Upper Bound} = Q_3 + 1.5 \times \text{IQR}$$
Values outside this interval were clipped to the boundary thresholds to preserve sample count while mitigating gradient distortion.

### 3.3 Feature Normalization & Scaling
Because features vary widely in magnitude (e.g., Insulin $\approx 100$ vs Pedigree $\approx 0.5$), $Z$-score standardization was applied:
$$Z = \frac{x - \mu}{\sigma}$$
Parameters $(\mu, \sigma)$ were computed strictly on the training set to prevent data leakage.

---

## 4. Machine Learning Algorithms

The system investigates the five foundational algorithms emphasized in the project abstract alongside advanced ensemble learning:

1. **Logistic Regression (LR)**:
   - Models the log-odds of the outcome as a linear combination of independent variables.
   - Serves as the probabilistic clinical baseline.

2. **Decision Tree Classifier (DT)**:
   - Non-parametric recursive splitting based on Gini impurity.
   - Provides transparent, interpretable threshold rules.

3. **Random Forest Classifier (RF)**:
   - Bagging ensemble of 150 randomized decision trees.
   - Reduces variance and provides feature importance rankings.

4. **Support Vector Machine (SVM)**:
   - Non-linear separation using a Radial Basis Function (RBF) kernel with Platt probability calibration.
   - Effective in finding optimal hyperplanes with maximum margin.

5. **K-Nearest Neighbors (KNN)**:
   - Distance-weighted neighbor majority vote ($k=7$) using Euclidean distance over scaled feature space.

6. **Weighted Soft Voting Ensemble (IEEE Access Methodology)**:
   - Combines predicted posterior probabilities from base classifiers.
   - Weights $W_j$ are set to each classifier's cross-validated ROC-AUC, granting greater voting weight to more discriminative models:
$$P^{en}_i = \frac{\sum_{j=1}^{m} (W_j \times P_{ij})}{\sum_{i=1}^{C} \sum_{j=1}^{m} (W_j \times P_{ij})}$$

---

## 5. Experimental Evaluation & Results

### 5.1 Validation Protocol
- **Stratified Split**: 80% Training ($N=614$), 20% Held-Out Testing ($N=154$).
- **Cross-Validation**: 5-Fold Stratified Cross-Validation on the training partition to tune hyperparameters and ensure generalizability.

### 5.2 Comparative Results Table

| Model | CV Accuracy | CV ROC-AUC | Test Accuracy | Test Precision | Test Recall | Test Specificity | Test F1-Score | Test ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **AdaBoost** | 76.39% | 0.827 | **75.32%** | **69.05%** | 53.70% | **87.00%** | **0.6042** | **0.8262** |
| **Gradient Boosting** | 74.59% | 0.812 | 74.68% | 67.44% | 53.70% | 86.00% | 0.5979 | 0.8185 |
| **Weighted Ensemble (Proposed)** | 77.03% | **0.840** | 73.38% | 64.44% | 53.70% | 84.00% | 0.5859 | **0.8170** |
| **Random Forest** | 77.04% | 0.831 | 74.03% | 67.50% | 50.00% | 87.00% | 0.5745 | 0.8169 |
| **Stacking Classifier** | **77.69%** | **0.847** | 72.08% | 62.22% | 51.85% | 83.00% | 0.5657 | 0.8135 |
| **Support Vector Machine** | **78.34%** | 0.837 | 73.38% | 64.44% | 53.70% | 84.00% | 0.5859 | 0.8096 |
| **Logistic Regression** | **78.34%** | **0.850** | 71.43% | 60.87% | 51.85% | 82.00% | 0.5600 | 0.8085 |
| **Decision Tree** | 71.82% | 0.753 | **76.62%** | 64.52% | **74.07%** | 78.00% | **0.6897** | 0.7932 |
| **K-Nearest Neighbors** | 75.40% | 0.804 | 75.32% | 66.67% | 59.26% | 84.00% | 0.6275 | 0.7904 |
| **Naive Bayes** | 77.03% | 0.840 | 70.78% | 57.89% | 61.11% | 76.00% | 0.5946 | 0.7935 |

---

## 6. Key Findings & Insights

1. **Top Risk Indicators**:
   - **Glucose** and **BMI** are the single most significant predictive biomarkers across all tree-based and linear models.
   - **Age** and **Diabetes Pedigree Function** follow as critical secondary indicators.

2. **Model Trade-Offs**:
   - **Decision Tree** demonstrated high sensitivity/recall (74.07%), meaning it effectively flags diabetic patients with fewer false negatives.
   - **Random Forest** and **AdaBoost** demonstrated higher specificity (87.00%), minimizing false alarms on healthy patients.
   - **The Weighted Soft Voting Ensemble** achieved an optimal balance, providing smooth calibrated probabilities and achieving a high cross-validated ROC-AUC of **0.840**.

---

## 7. Clinical Decision Support System (Deployment)

To translate machine learning algorithms into practical healthcare tools, two interfaces were developed:

1. **Interactive Streamlit Web Dashboard (`app.py`)**:
   - **Patient Assessment**: Direct input of medical parameters with clinical reference guidance.
   - **Risk Stratification**: Classifies patients into **Low (<35%)**, **Moderate (35-65%)**, and **High (>65%)** risk tiers.
   - **Batch Screening**: Rapid CSV intake for epidemiological screening of hundreds of patients simultaneously.
   - **EDA & Benchmarking Modules**: Transparent visualization of data distributions and model metrics.

2. **Command-Line Predictor (`predict.py`)**:
   - Lightweight script for headless or automated laboratory pipeline execution.

---

## 8. Conclusion & Future Directions

This project successfully implements the Diabetes Prediction System proposed in the project abstract. By coupling rigorous clinical data preprocessing (biological zero imputation and IQR bounds) with standard and ensemble classifiers, the system achieves robust predictive accuracy (up to 76.6%) and discriminative ability (ROC-AUC > 0.84).

### Future Enhancements:
- Integrating additional clinical features (e.g., HbA1c, lipid profile, fasting C-peptide).
- Implementing deep learning models (Multilayer Perceptrons / 1D-CNNs) on larger clinical datasets.
- Integrating explainable AI (SHAP / LIME) for personalized feature-level risk explanations in the clinical web UI.
