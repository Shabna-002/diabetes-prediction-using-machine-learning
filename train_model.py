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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
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

# Define the evaluated algorithms including KNN, SVM, and Gradient Boosting
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
    ),
    "K-Nearest Neighbours (KNN)": (
        "knn",
        KNeighborsClassifier(n_neighbors=5)
    ),
    "Gradient Boosting": (
        "gradient_boosting",
        GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
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

# ------------------------------------------------------------------
# Generate docs/models_data.js for GitHub Pages Client-side Inference
# ------------------------------------------------------------------
def export_web_models():
    DOCS_DIR = os.path.join(BASE_DIR, "docs")
    if not os.path.exists(DOCS_DIR):
        return

    # Extract fitted preprocessing from best pipeline
    imputer = best_pipeline.named_steps["imputer"]
    scaler = best_pipeline.named_steps["scaler"]
    
    imputer_stats = [round(float(x), 4) for x in imputer.statistics_]
    scaler_mean = [round(float(x), 5) for x in scaler.mean_]
    scaler_scale = [round(float(x), 5) for x in scaler.scale_]

    # Pre-transform training data for KNN
    X_train_imp = imputer.transform(X_train)
    X_train_scaled = scaler.transform(X_train_imp)

    # 1. Decision Tree serialization
    dt_pipeline = joblib.load(os.path.join(MODEL_DIR, "decision_tree.joblib"))
    dt_clf = dt_pipeline.named_steps["classifier"]
    
    def serialize_tree(tree):
        def recurse(node):
            if tree.children_left[node] == -1 and tree.children_right[node] == -1:
                val = tree.value[node][0]
                tot = val.sum()
                prob = val[1] / tot if tot > 0 else 0.0
                return {"leaf": 1, "pred": int(prob >= 0.5), "prob": round(float(prob), 4)}
            return {
                "f": int(tree.feature[node]),
                "th": round(float(tree.threshold[node]), 5),
                "l": recurse(tree.children_left[node]),
                "r": recurse(tree.children_right[node])
            }
        return recurse(0)

    decision_tree_data = serialize_tree(dt_clf.tree_)

    # 2. Logistic Regression
    lr_pipeline = joblib.load(os.path.join(MODEL_DIR, "logistic_regression.joblib"))
    lr_clf = lr_pipeline.named_steps["classifier"]
    logistic_data = {
        "coef": [round(float(c), 5) for c in lr_clf.coef_[0]],
        "intercept": round(float(lr_clf.intercept_[0]), 5)
    }

    # 3. Random Forest (all trees)
    rf_pipeline = joblib.load(os.path.join(MODEL_DIR, "random_forest.joblib"))
    rf_clf = rf_pipeline.named_steps["classifier"]
    rf_trees = [serialize_tree(est.tree_) for est in rf_clf.estimators_]

    # 4. K-Nearest Neighbours (KNN)
    knn_data = {
        "k": 5,
        "X_train": [[round(float(v), 4) for v in row] for row in X_train_scaled],
        "y_train": [int(v) for v in y_train]
    }

    # 5. Support Vector Machine (SVM)
    svm_pipeline = joblib.load(os.path.join(MODEL_DIR, "svm.joblib"))
    svm_clf = svm_pipeline.named_steps["classifier"]
    gamma_val = float(svm_clf._gamma) if hasattr(svm_clf, "_gamma") else 1.0 / (len(FEATURES) * X_train_scaled.var())
    svm_data = {
        "gamma": round(gamma_val, 6),
        "intercept": round(float(svm_clf.intercept_[0]), 5),
        "dual_coef": [round(float(c), 5) for c in svm_clf.dual_coef_[0]],
        "support_vectors": [[round(float(v), 4) for v in row] for row in svm_clf.support_vectors_]
    }

    # 6. Gradient Boosting
    gb_pipeline = joblib.load(os.path.join(MODEL_DIR, "gradient_boosting.joblib"))
    gb_clf = gb_pipeline.named_steps["classifier"]
    
    def serialize_gb_tree(tree):
        def recurse(node):
            if tree.children_left[node] == -1 and tree.children_right[node] == -1:
                return {"leaf": 1, "val": round(float(tree.value[node, 0, 0]), 5)}
            return {
                "f": int(tree.feature[node]),
                "th": round(float(tree.threshold[node]), 5),
                "l": recurse(tree.children_left[node]),
                "r": recurse(tree.children_right[node])
            }
        return recurse(0)

    raw_init = float(gb_clf._raw_predict_init(X_train[:1])[0][0])
    gb_trees = [serialize_gb_tree(est[0].tree_) for est in gb_clf.estimators_]
    gb_data = {
        "learning_rate": round(float(gb_clf.learning_rate), 4),
        "init_val": round(raw_init, 5),
        "trees": gb_trees
    }

    models_payload = {
        "features": FEATURES,
        "imputer_stats": imputer_stats,
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
        "decision_tree": decision_tree_data,
        "logistic_regression": logistic_data,
        "random_forest_trees": rf_trees,
        "knn": knn_data,
        "svm": svm_data,
        "gradient_boosting": gb_data,
        "metrics": metrics_payload
    }

    js_content = f"window.DIABETES_ML_MODELS = {json.dumps(models_payload, separators=(',', ':'))};"
    js_path = os.path.join(DOCS_DIR, "models_data.js")
    with open(js_path, "w", encoding="utf-8") as f:
        f.write(js_content)
    print(f"[*] Exported all 6 browser-compatible models to: docs/models_data.js ({len(js_content)} bytes)")

export_web_models()
