# Diabetes Prediction Using Machine Learning (DiaPredict)

> 🌐 **Live Web Application Demo**: [https://shabna-002.github.io/diabetes-prediction-using-machine-learning/docs/](https://shabna-002.github.io/diabetes-prediction-using-machine-learning/docs/)  
> 📊 **Clinical Dataset (CSV)**: [diabetes.csv](https://github.com/Shabna-002/diabetes-prediction-using-machine-learning/blob/main/diabetes.csv) | [Direct Download (Raw CSV)](https://raw.githubusercontent.com/Shabna-002/diabetes-prediction-using-machine-learning/main/diabetes.csv)

An end-to-end Machine Learning clinical screening web application training, benchmarking, and deploying **Logistic Regression**, **Decision Tree**, **Random Forest**, **K-Nearest Neighbours (KNN)**, **Support Vector Machine (SVM)**, and **Naive Bayes (GaussianNB)** for Type-2 Diabetes risk screening.

---

## 🌟 Key Methodological Highlights

1. **Leak-Free Scikit-Learn Pipelines**:
   - Preprocessing steps (`SimpleImputer` with median strategy and `StandardScaler`) are encapsulated within Scikit-learn `Pipeline` objects.
   - All statistics (medians, means, standard deviations) are fitted **strictly on the training partition/folds**, completely preventing test-set data leakage.
2. **Biological Missing Value Imputation**:
   - Biologically implausible zero measurements in physiological features (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) are marked as missing (`NaN`) and imputed with robust training medians.
3. **Identical Stratified Split**:
   - 80/20 train-test partition (`random_state=42`, stratified on `Outcome`), ensuring identical training and test data across all evaluated classifiers.
4. **Clinical Model Selection via 5-Fold Cross-Validation**:
   - **No Test Set Snooping**: The winning model is selected **strictly via 5-Fold Stratified Cross-Validation on the training set**. The test set is completely held out and never used for model selection.
   - **Recall Prioritization via $F_2$-Score**: In clinical diabetes screening, a **False Negative** (missing a diabetic patient) delays essential lifestyle/pharmacological intervention, causing irreversible vascular or organ damage. Conversely, a **False Positive** only leads to routine confirmatory lab tests (HbA1c/OGTT). Model selection optimizes the **$F_2$-score ($\beta=2$)**, weighting Recall twice as heavily as Precision.
5. **Multi-Metric Evaluation & Interactive Visualizations**:
   - Evaluated using **Accuracy, Precision, Recall, F1-score, F2-score, ROC-AUC**, 2x2 confusion matrices, and multi-model ROC curves.
   - Interactive charts rendered via **Chart.js** on the web interface.
6. **Dual Database Architecture**:
   - Native **MySQL** connectivity for enterprise clinical audit logging.
   - Graceful, automatic **SQLite fallback** (`data/predictions.db`) for instant zero-configuration local runs.
7. **User Authentication & Demo Accounts**:
   - Secure password hashing using PBKDF2 (`werkzeug.security`).
   - Pre-seeded demo credentials: `admin` / `admin123` and `doctor` / `doctor123`.

---

## 📊 Measured Benchmark Performance (Empirical Results)

### Phase 1: 5-Fold Stratified Cross-Validation on Training Set (Model Selection)
*Metric used for selection: Mean CV $F_2$-Score (prioritizing Recall while maintaining Precision & F1)*

| Classifier | Mean CV Recall | Mean CV Precision | Mean CV F1 | Mean CV $F_2$ (Selection) | Mean CV ROC-AUC | Mean CV Accuracy | Selection Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Naive Bayes (GaussianNB)** | **0.5984** | 0.6859 | 0.6369 | **0.6128** | 0.8281 | 76.23% | **★ Clinically Selected Winner** |
| **Support Vector Machine (SVM)** | 0.5842 | 0.7415 | 0.6485 | 0.6074 | 0.8333 | 78.01% | Evaluated |
| **Logistic Regression** | 0.5748 | 0.7638 | 0.6545 | 0.6040 | 0.8434 | 78.82% | Evaluated |
| **Random Forest (100 Trees)** | 0.5608 | 0.7329 | 0.6328 | 0.5871 | 0.8351 | 77.20% | Evaluated |
| **K-Nearest Neighbours (KNN, k=5)**| 0.5654 | 0.6367 | 0.5939 | 0.5755 | 0.7843 | 72.96% | Evaluated |
| **Decision Tree (max_depth=5)** | 0.5463 | 0.6203 | 0.5763 | 0.5571 | 0.7655 | 72.48% | Evaluated |
| *Gradient Boosting (Auxiliary)* | 0.5981 | 0.6693 | 0.6293 | 0.6098 | 0.8205 | 75.41% | Evaluated |

### Phase 2: Independent Held-Out Test Set Evaluation (154 Unseen Samples)
*Unbiased generalization metrics evaluated after model selection:*

| Classifier | Test Accuracy | Test Precision | Test Recall (Sensitivity) | Test F1-Score | Test $F_2$-Score | Test ROC-AUC | Confusion Matrix `[TN, FP / FN, TP]` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Naive Bayes** | 70.13% | 0.5667 | **0.6296** | 0.5965 | **0.6159** | 0.7646 | `[74, 26 / 20, 34]` |
| **Decision Tree** | 76.62% | 0.6500 | 0.7222 | 0.6842 | 0.7065 | 0.7840 | `[79, 21 / 15, 39]` |
| **Random Forest** | 74.03% | 0.6667 | 0.5185 | 0.5833 | 0.5426 | **0.8167** | `[86, 14 / 26, 28]` |
| **K-Nearest Neighbours (KNN)**| 75.32% | 0.6600 | 0.6111 | 0.6346 | 0.6203 | 0.7899 | `[83, 17 / 21, 33]` |
| **Support Vector Machine (SVM)**| 74.03% | 0.6522 | 0.5556 | 0.6000 | 0.5725 | 0.7964 | `[84, 16 / 24, 30]` |
| **Logistic Regression** | 70.78% | 0.6000 | 0.5000 | 0.5455 | 0.5172 | **0.8130** | `[82, 18 / 27, 27]` |
| *Gradient Boosting* | 75.97% | 0.6889 | 0.5741 | 0.6263 | 0.5939 | 0.8306 | `[86, 14 / 23, 31]` |

---

## 📁 Project Architecture

```
diabetes_prediction_ml_project/
├── app.py                      # Flask web application & dual database routes
├── train_model.py              # Reproducible training & multi-classifier benchmark script
├── requirements.txt            # Python dependencies (Flask, scikit-learn, pandas, etc.)
├── database.sql                # MySQL schema and indexes
├── diabetes.csv                # Primary 768-sample clinical dataset
├── data/
│   ├── diabetes.csv            # Data copy
│   └── predictions.db          # Auto-generated SQLite audit log database
├── model/
│   ├── diabetes_model.joblib   # Selected inference pipeline (Naive Bayes)
│   ├── naive_bayes.joblib      # Naive Bayes pipeline
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── random_forest.joblib
│   ├── knn.joblib
│   ├── svm.joblib
│   ├── gradient_boosting.joblib
│   └── model_metrics.json      # Benchmark results, CV scores, & ROC curve points
├── templates/                  # Jinja2 HTML templates
│   ├── base.html               # Shared layout, navbar, Chart.js loader, user badge
│   ├── index.html              # Hero presentation & pipeline highlights
│   ├── models.html             # Benchmark table, Chart.js bar & ROC charts, confusion matrices
│   ├── predict.html            # Patient risk assessment form & algorithm picker
│   ├── result.html             # Risk score & probability display
│   ├── dashboard.html          # Storage & outcome statistics
│   ├── history.html            # Prediction logs with print & delete
│   ├── login.html              # User sign in with demo shortcuts
│   └── register.html           # New account registration
├── static/
│   ├── css/style.css           # Glassmorphism cyber-medical styling
│   └── js/
│       ├── app.js              # Client-side input validation
│       └── chart.min.js        # Local Chart.js library fallback
├── tests/
│   └── test_full_suite.py      # Automated unit tests covering all routes and models
└── docs/                       # Static demo & academic documentation
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.10+ installed on your system.

### 2. Install Dependencies
```powershell
py -3.10 -m pip install -r requirements.txt
```

### 3. Run Reproducible Model Training & Benchmarking
To retrain all 6+ classifiers, execute 5-fold cross-validation on the training set, optimize the $F_2$-score, and regenerate all model pipelines:
```powershell
py -3.10 train_model.py
```
This script will:
- Mark invalid zeros as missing for `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`.
- Construct leak-free Scikit-learn `Pipeline` objects (`SimpleImputer` + `StandardScaler` + Classifier).
- Perform 5-Fold Stratified Cross-Validation on `X_train` to select the winning screening model ($F_2$-Score).
- Fit pipelines on `X_train` and evaluate on held-out `X_test`.
- Save `model/diabetes_model.joblib`, individual `.joblib` files, and `model/model_metrics.json`.

### 4. Run the Full Test Suite
Verify that all routes, models, and features work properly:
```powershell
py -3.10 -m unittest tests/test_full_suite.py
```

### 5. Launch the Web Application
```powershell
py -3.10 app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

### 6. Demo User Credentials
You can log in at `/login` using the pre-seeded demo accounts:
- **Admin**: `admin` / `admin123`
- **Clinician**: `doctor` / `doctor123`
- Or register a new account on `/register`.

---

## 🔄 Retraining Instructions When Dataset Changes

When a new or updated diabetes dataset is provided:
1. Replace `diabetes.csv` in the root (or `data/diabetes.csv`) with the updated CSV file. Ensure the standard 8 clinical features are present:
   - `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`, and target `Outcome`.
2. Run the automated retraining script:
   ```powershell
   py -3.10 train_model.py
   ```
3. The script will:
   - Compute new median statistics and standard scaling parameters on the updated training partition.
   - Run 5-fold cross-validation on the training set to identify the new best model by $F_2$-Score.
   - Evaluate all models on the held-out test set and export updated ROC curves and confusion matrices.
   - Overwrite `model/diabetes_model.joblib` and `model/model_metrics.json`.
4. Restart or refresh `app.py`. The web application will immediately use the newly trained pipeline.

---

## 🗄️ Database Configuration (MySQL & SQLite)

The application supports dual database storage:
- **MySQL**: Connects using credentials defined in `app.py` or environment variables:
  - `DB_HOST` (default: `127.0.0.1`)
  - `DB_USER` (default: `root`)
  - `DB_PASSWORD` (default: `""`)
  - `DB_NAME` (default: `diabetes_prediction_db`)
  - You can import `database.sql` into your MySQL instance: `mysql -u root -p < database.sql`
- **SQLite Fallback**: If MySQL is unreachable or disabled (`USE_SQLITE_ONLY=1`), the application seamlessly uses SQLite (`data/predictions.db`) without throwing errors or requiring any configuration.

---

## ⚠️ Clinical Disclaimer
DiaPredict is an educational and academic demonstration. The predictions generated by these statistical models are intended for educational screening and should never replace professional medical diagnosis, laboratory testing, or clinical consultation.
