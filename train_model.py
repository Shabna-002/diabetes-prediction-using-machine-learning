import os
import json
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "diabetes.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]
TARGET = "Outcome"

# Load dataset
df = pd.read_csv(DATA_PATH)

# In this dataset, zero is biologically invalid for several measurements.
# Treat those zeros as missing values and impute them during training.
zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[zero_as_missing] = df[zero_as_missing].replace(0, pd.NA)

X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
y = df[TARGET].astype(int)

# Stratified 80/20 train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Define the 4 algorithms specified in the project abstract
classifiers = {
    "Logistic Regression": (
        "logistic_regression",
        LogisticRegression(max_iter=1000, random_state=42)
    ),
    "Decision Tree": (
        "decision_tree",
        DecisionTreeClassifier(max_depth=5, min_samples_split=5, random_state=42)
    ),
    "Random Forest": (
        "random_forest",
        RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    ),
    "Support Vector Machine (SVM)": (
        "svm",
        SVC(kernel="rbf", C=1.0, probability=True, random_state=42)
    )
}

results = {}
best_model_name = None
best_f1 = -1.0
best_pipeline = None

print("==================================================================")
print("     DIABETES PREDICTION USING MACHINE LEARNING - BENCHMARK      ")
print("==================================================================")
print(f"Total dataset size: {len(df)} records | Training: {len(X_train)} | Test: {len(X_test)}")
print("Pre-processing: Median Imputation + StandardScaler")
print("------------------------------------------------------------------\n")

for name, (slug, clf) in classifiers.items():
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", clf)
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    # Save individual model pipeline
    model_file = os.path.join(MODEL_DIR, f"{slug}.joblib")
    joblib.dump(pipeline, model_file)
    
    results[name] = {
        "slug": slug,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm,
        "model_file": f"{slug}.joblib"
    }
    
    print(f"Algorithm: {name}")
    print(f"  Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-score : {f1:.4f}")
    print(f"  Confusion Matrix: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("-" * 60)
    
    # Select best model primarily by F1-score (or accuracy if tied)
    score_metric = f1 * 100 + acc
    if score_metric > best_f1:
        best_f1 = score_metric
        best_model_name = name
        best_pipeline = pipeline

# Save the best model as diabetes_model.joblib for default application use
joblib.dump(best_pipeline, os.path.join(MODEL_DIR, "diabetes_model.joblib"))

# Prepare metrics artifact for the web app
metrics_payload = {
    "dataset_info": {
        "total_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "features": FEATURES
    },
    "best_model": best_model_name,
    "models": results
}

metrics_path = os.path.join(MODEL_DIR, "model_metrics.json")
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics_payload, f, indent=2)

print("\n==================================================================")
print(f"[BEST PERFORMING MODEL]: {best_model_name}")
print(f"[*] Saved best pipeline to: model/diabetes_model.joblib")
print(f"[*] Saved benchmark metrics to: model/model_metrics.json")
print("==================================================================")
