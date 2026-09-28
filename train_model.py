"""
=============================================================================
DiaPredict - Complete Model Training and Classifier Comparison Module
=============================================================================
Reproducible Machine Learning pipeline for Type-2 Diabetes Risk Prediction.

Key Methodological Standards:
1. Biological Cleaning: Replaces invalid 0 values with NaN for physiological
   features (Glucose, Blood Pressure, Skin Thickness, Insulin, BMI).
2. Leakage-Free Preprocessing: Uses Scikit-learn Pipeline with SimpleImputer
   (median strategy) and StandardScaler fitted ONLY on training folds/data.
3. Identical Stratified Split: 80/20 train-test split stratified on Outcome with
   fixed random_state=42, ensuring identical training and test data for all models.
4. Comprehensive Classifier Suite:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - K-Nearest Neighbours (KNN)
   - Support Vector Machine (SVM)
   - Naive Bayes (GaussianNB)
   - Gradient Boosting (Auxiliary benchmark)
5. Clinical Model Selection:
   - Model selection is performed STRICTLY via 5-fold Stratified Cross-Validation
     on the training set (X_train, y_train).
   - The primary selection metric is the F2-Score (beta=2), which prioritizes
     Recall (Sensitivity) to minimize life-threatening False Negatives, while
     maintaining Precision and F1-score balance.
   - The test set (X_test, y_test) is strictly held out and NEVER used for model
     selection, preventing data snooping and optimistic bias.
6. Persistence:
   - Winning pipeline saved to `model/diabetes_model.joblib`.
   - All individual model pipelines saved to `model/<slug>.joblib`.
   - Detailed benchmark metrics & ROC coordinates saved to `model/model_metrics.json`.
   - Browser inference models exported to `docs/models_data.js`.
=============================================================================
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    fbeta_score, roc_auc_score, confusion_matrix, roc_curve,
    classification_report, make_scorer
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "diabetes.csv")
if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join(BASE_DIR, "diabetes.csv")

MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]
TARGET = "Outcome"

# Features where 0 is physiologically invalid and represents missing data
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_and_clean_data(filepath=DATA_PATH):
    """Loads dataset and marks physiologically impossible zeros as NaN."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    df = pd.read_csv(filepath)
    df[ZERO_AS_MISSING] = df[ZERO_AS_MISSING].replace(0, np.nan)
    
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    y = df[TARGET].astype(int)
    return df, X, y


def build_pipeline(classifier):
    """Constructs a leak-free Scikit-learn Pipeline with median imputation and scaling."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", classifier)
    ])


def downsample_roc(fpr, tpr, max_points=35):
    """Downsamples ROC curve coordinates for clean, fast web charting."""
    if len(fpr) <= max_points:
        return [{"x": round(float(x), 4), "y": round(float(y), 4)} for x, y in zip(fpr, tpr)]
    
    indices = np.linspace(0, len(fpr) - 1, max_points, dtype=int)
    # Ensure (0,0) and (1,1) endpoints are included
    selected_indices = sorted(list(set([0] + list(indices) + [len(fpr) - 1])))
    return [{"x": round(float(fpr[i]), 4), "y": round(float(tpr[i]), 4)} for i in selected_indices]


def train_and_evaluate_all():
    print("=" * 76)
    print("      DIAPREDICT - MULTI-CLASSIFIER BENCHMARK & CLINICAL SELECTION     ")
    print("=" * 76)

    df, X, y = load_and_clean_data()
    print(f"Dataset Loaded: {len(df)} records from {DATA_PATH}")
    print(f"Features ({len(FEATURES)}): {', '.join(FEATURES)}")
    print(f"Target: {TARGET} (Class 0: {(y==0).sum()}, Class 1: {(y==1).sum()})")
    print("Physiological Missing Value Handling: Median Imputation on Glucose, BP, SkinThickness, Insulin, BMI")
    print("Feature Standardization: StandardScaler fitted strictly on training data")
    print("-" * 76)

    # 1. Stratified 80/20 train-test split with fixed random_state
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training Samples: {len(X_train)} | Stratified Test Samples: {len(X_test)}")
    print("-" * 76)

    # 2. Define the candidate classifiers
    classifiers = {
        "Logistic Regression": {
            "slug": "logistic_regression",
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "description": "Linear classifier optimizing log-odds with L2 regularization."
        },
        "Decision Tree": {
            "slug": "decision_tree",
            "model": DecisionTreeClassifier(max_depth=5, min_samples_split=5, random_state=42),
            "description": "Non-linear decision boundary tree with depth constraints to control overfitting."
        },
        "Random Forest": {
            "slug": "random_forest",
            "model": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
            "description": "Ensemble of 100 bagged trees with bootstrap aggregation and feature subsampling."
        },
        "K-Nearest Neighbours (KNN)": {
            "slug": "knn",
            "model": KNeighborsClassifier(n_neighbors=5),
            "description": "Instance-based non-parametric classifier using standardized Euclidean distance."
        },
        "Support Vector Machine (SVM)": {
            "slug": "svm",
            "model": SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
            "description": "Kernelized maximum-margin hyperplane classifier with Radial Basis Function (RBF)."
        },
        "Naive Bayes": {
            "slug": "naive_bayes",
            "model": GaussianNB(),
            "description": "Probabilistic Gaussian classifier assuming conditional feature independence."
        },
        "Gradient Boosting": {
            "slug": "gradient_boosting",
            "model": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
            "description": "Sequential boosting ensemble minimizing binomial deviance loss."
        }
    }

    # 3. 5-Fold Stratified Cross-Validation on TRAINING SET ONLY for Model Selection
    f2_scorer = make_scorer(fbeta_score, beta=2, zero_division=0)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "f2": f2_scorer,
        "roc_auc": "roc_auc"
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n>>> PHASE 1: 5-FOLD CROSS-VALIDATION ON TRAINING SET (MODEL SELECTION)")
    print("    Objective: Select model maximizing clinical screening utility (F2-Score: Recall priority)")
    print("    (Note: Held-out test set is STRICTLY untouched during this phase to prevent data leakage)\n")

    cv_results = {}
    fitted_pipelines = {}

    for name, config in classifiers.items():
        pipe = build_pipeline(config["model"])
        cv_out = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, return_train_score=False)
        
        cv_summary = {
            "accuracy": float(cv_out["test_accuracy"].mean()),
            "precision": float(cv_out["test_precision"].mean()),
            "recall": float(cv_out["test_recall"].mean()),
            "f1_score": float(cv_out["test_f1"].mean()),
            "f2_score": float(cv_out["test_f2"].mean()),
            "roc_auc": float(cv_out["test_roc_auc"].mean()),
            "accuracy_std": float(cv_out["test_accuracy"].std()),
            "recall_std": float(cv_out["test_recall"].std()),
            "f2_std": float(cv_out["test_f2"].std())
        }
        cv_results[name] = cv_summary
        
        print(f"[{name}]")
        print(f"  CV Recall (Sensitivity) : {cv_summary['recall']:.4f} (+/- {cv_summary['recall_std']:.4f})")
        print(f"  CV Precision            : {cv_summary['precision']:.4f}")
        print(f"  CV F1-Score             : {cv_summary['f1_score']:.4f}")
        print(f"  CV F2-Score (Selection) : {cv_summary['f2_score']:.4f} (+/- {cv_summary['f2_std']:.4f})")
        print(f"  CV ROC-AUC              : {cv_summary['roc_auc']:.4f}")
        print(f"  CV Accuracy             : {cv_summary['accuracy']:.4f}")
        print("-" * 55)

    # Select best model strictly from Training CV F2-Score
    best_model_name = max(cv_results.keys(), key=lambda k: cv_results[k]["f2_score"])
    best_cv_f2 = cv_results[best_model_name]["f2_score"]
    best_cv_recall = cv_results[best_model_name]["recall"]

    print("\n" + "=" * 76)
    print(f"  >>> CLINICALLY SELECTED MODEL: {best_model_name}")
    print(f"      Selected strictly via Training 5-Fold Cross-Validation F2-Score: {best_cv_f2:.4f}")
    print(f"      Selected Training CV Recall (Sensitivity): {best_cv_recall:.4f}")
    print("      Clinical Rationale: F2-score places double weight on Recall over Precision,")
    print("      minimizing dangerous False Negatives in asymptomatic diabetes screening.")
    print("=" * 76 + "\n")

    # 4. Fit all pipelines on full training set and evaluate on UNTOUCHED TEST SET
    print(">>> PHASE 2: INDEPENDENT HELD-OUT TEST SET EVALUATION (UNBIASED GENERALIZATION)")
    print(f"    Evaluating on {len(X_test)} unseen test samples...\n")

    test_results = {}
    best_pipeline = None

    for name, config in classifiers.items():
        slug = config["slug"]
        pipe = build_pipeline(config["model"])
        
        # Fit strictly on X_train (imputer and scaler learn statistics only from training data)
        pipe.fit(X_train, y_train)
        fitted_pipelines[slug] = pipe

        y_pred = pipe.predict(X_test)
        
        if hasattr(pipe, "predict_proba"):
            y_proba = pipe.predict_proba(X_test)[:, 1]
        elif hasattr(pipe, "decision_function"):
            df_vals = pipe.decision_function(X_test)
            y_proba = 1.0 / (1.0 + np.exp(-df_vals))
        else:
            y_proba = y_pred.astype(float)

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        f2 = float(fbeta_score(y_test, y_pred, beta=2, zero_division=0))
        auc = float(roc_auc_score(y_test, y_proba))
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_pts = downsample_roc(fpr, tpr)

        # Save individual model pipeline
        model_file = os.path.join(MODEL_DIR, f"{slug}.joblib")
        joblib.dump(pipe, model_file)

        test_results[name] = {
            "slug": slug,
            "description": config["description"],
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "f2_score": round(f2, 4),
            "roc_auc": round(auc, 4),
            "confusion_matrix": cm,
            "roc_curve": roc_pts,
            "cv_metrics": {
                "cv_accuracy": round(cv_results[name]["accuracy"], 4),
                "cv_precision": round(cv_results[name]["precision"], 4),
                "cv_recall": round(cv_results[name]["recall"], 4),
                "cv_f1_score": round(cv_results[name]["f1_score"], 4),
                "cv_f2_score": round(cv_results[name]["f2_score"], 4),
                "cv_roc_auc": round(cv_results[name]["roc_auc"], 4),
            },
            "model_file": f"{slug}.joblib"
        }

        if name == best_model_name:
            best_pipeline = pipe

        print(f"[{name}]")
        print(f"  Test Accuracy : {acc:.4f} ({acc*100:.2f}%)")
        print(f"  Test Precision: {prec:.4f}")
        print(f"  Test Recall   : {rec:.4f} (Sensitivity)")
        print(f"  Test F1-Score : {f1:.4f}")
        print(f"  Test F2-Score : {f2:.4f}")
        print(f"  Test ROC-AUC  : {auc:.4f}")
        print(f"  Confusion Matrix: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")
        print(f"  Saved pipeline: model/{slug}.joblib")
        print("-" * 55)

    # 5. Save the winning pipeline as default diabetes_model.joblib
    diabetes_model_path = os.path.join(MODEL_DIR, "diabetes_model.joblib")
    joblib.dump(best_pipeline, diabetes_model_path)
    print(f"\n[*] Default inference pipeline saved to: {diabetes_model_path}")

    # 6. Save comprehensive benchmark metadata artifact
    metrics_payload = {
        "dataset_info": {
            "total_samples": len(df),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "features": FEATURES,
            "zero_as_missing": ZERO_AS_MISSING,
            "imputation_strategy": "median",
            "scaling": "StandardScaler (mean=0, variance=1)"
        },
        "model_selection": {
            "selection_strategy": "5-Fold Stratified Cross-Validation on Training Set",
            "selection_metric": "F2-Score (beta=2.0)",
            "test_set_used_for_selection": False,
            "rationale": (
                "In clinical diabetes screening, failing to detect a diabetic patient (False Negative) "
                "imposes severe diagnostic risk, causing untreated disease progression. The F2-score "
                "prioritizes Recall (Sensitivity) with twice the weight of Precision, while maintaining "
                "sound specificity. Cross-validation on the training set ensures the selection is completely "
                "unbiased by test set data."
            ),
            "best_model": best_model_name,
            "best_cv_f2": round(best_cv_f2, 4),
            "best_cv_recall": round(best_cv_recall, 4)
        },
        "best_model": best_model_name,
        "models": test_results
    }

    metrics_json_path = os.path.join(MODEL_DIR, "model_metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"[*] Benchmark metrics artifact saved to: {metrics_json_path}")

    # 7. Export client-side models for docs / static demo
    export_web_models(fitted_pipelines, X_train, y_train, metrics_payload)

    print("\n==================================================================")
    print(" [SUCCESS] BENCHMARK COMPLETE: ALL 6+ CLASSIFIERS TRAINED & PERSISTED")
    print("==================================================================")
    return metrics_payload


def export_web_models(fitted_pipelines, X_train, y_train, metrics_payload):
    """Exports browser-compatible JSON model weights for docs/models_data.js."""
    DOCS_DIR = os.path.join(BASE_DIR, "docs")
    if not os.path.exists(DOCS_DIR):
        return

    best_pipe = fitted_pipelines.get(
        metrics_payload["models"][metrics_payload["best_model"]]["slug"],
        list(fitted_pipelines.values())[0]
    )
    imputer = best_pipe.named_steps["imputer"]
    scaler = best_pipe.named_steps["scaler"]

    imputer_stats = [round(float(x), 4) for x in imputer.statistics_]
    scaler_mean = [round(float(x), 5) for x in scaler.mean_]
    scaler_scale = [round(float(x), 5) for x in scaler.scale_]

    X_train_imp = imputer.transform(X_train)
    X_train_scaled = scaler.transform(X_train_imp)

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

    # 1. Decision Tree
    dt_clf = fitted_pipelines["decision_tree"].named_steps["classifier"]
    decision_tree_data = serialize_tree(dt_clf.tree_)

    # 2. Logistic Regression
    lr_clf = fitted_pipelines["logistic_regression"].named_steps["classifier"]
    logistic_data = {
        "coef": [round(float(c), 5) for c in lr_clf.coef_[0]],
        "intercept": round(float(lr_clf.intercept_[0]), 5)
    }

    # 3. Random Forest
    rf_clf = fitted_pipelines["random_forest"].named_steps["classifier"]
    rf_trees = [serialize_tree(est.tree_) for est in rf_clf.estimators_]

    # 4. KNN
    knn_data = {
        "k": 5,
        "X_train": [[round(float(v), 4) for v in row] for row in X_train_scaled],
        "y_train": [int(v) for v in y_train]
    }

    # 5. SVM
    svm_clf = fitted_pipelines["svm"].named_steps["classifier"]
    gamma_val = float(svm_clf._gamma) if hasattr(svm_clf, "_gamma") else 1.0 / (len(FEATURES) * X_train_scaled.var())
    svm_data = {
        "gamma": round(gamma_val, 6),
        "intercept": round(float(svm_clf.intercept_[0]), 5),
        "dual_coef": [round(float(c), 5) for c in svm_clf.dual_coef_[0]],
        "support_vectors": [[round(float(v), 4) for v in row] for row in svm_clf.support_vectors_]
    }

    # 6. Naive Bayes (GaussianNB)
    nb_clf = fitted_pipelines["naive_bayes"].named_steps["classifier"]
    nb_data = {
        "class_prior": [round(float(p), 5) for p in nb_clf.class_prior_],
        "theta": [[round(float(v), 5) for v in row] for row in nb_clf.theta_],
        "var": [[round(float(v), 5) for v in row] for row in nb_clf.var_]
    }

    # 7. Gradient Boosting
    gb_clf = fitted_pipelines["gradient_boosting"].named_steps["classifier"]
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

    raw_init = float(gb_clf._raw_predict_init(X_train_scaled[:1])[0][0])
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
        "naive_bayes": nb_data,
        "gradient_boosting": gb_data,
        "metrics": metrics_payload
    }

    js_content = f"window.DIABETES_ML_MODELS = {json.dumps(models_payload, separators=(',', ':'))};"
    js_path = os.path.join(DOCS_DIR, "models_data.js")
    with open(js_path, "w", encoding="utf-8") as f:
        f.write(js_content)
    print(f"[*] Exported browser models (including Naive Bayes) to: {js_path} ({len(js_content)} bytes)")


if __name__ == "__main__":
    train_and_evaluate_all()
