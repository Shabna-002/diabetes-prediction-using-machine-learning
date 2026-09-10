"""
Model Training and Ensembling Pipeline
Implements individual ML classifiers, 5-Fold Stratified Cross-Validation,
and the Weighted Soft Voting Ensemble formulated in:
"Diabetes Prediction Using Ensembling of Different Machine Learning Classifiers" (IEEE Access, 2020)
"""

import os
import sys
import json

# Ensure src directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    StackingClassifier,
    VotingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from data_preprocessing import DiabetesDataPipeline, FEATURE_NAMES

try:
    from config import MODELS_DIR, REPORTS_DIR, RANDOM_STATE, CV_FOLDS
except ImportError:
    try:
        from src.config import MODELS_DIR, REPORTS_DIR, RANDOM_STATE, CV_FOLDS
    except ImportError:
        MODELS_DIR = "models"
        REPORTS_DIR = "reports"
        RANDOM_STATE = 42
        CV_FOLDS = 5


def get_base_classifiers(random_state=RANDOM_STATE):
    """Returns a dictionary of candidate base machine learning classifiers."""
    models = {
        'Logistic Regression': LogisticRegression(
            C=1.0, max_iter=1000, random_state=random_state
        ),
        'Support Vector Machine': SVC(
            C=1.0, kernel='rbf', probability=True, random_state=random_state
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=5, min_samples_split=4, random_state=random_state
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=150, max_depth=6, random_state=random_state
        ),
        'K-Nearest Neighbors': KNeighborsClassifier(
            n_neighbors=7, weights='distance'
        ),
        'AdaBoost': AdaBoostClassifier(
            n_estimators=100, learning_rate=0.8, random_state=random_state
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.08, max_depth=3, random_state=random_state
        ),
        'Naive Bayes': GaussianNB()
    }
    return models


def evaluate_model_cv(model, X, y, cv=5):
    """Evaluates a classifier using Stratified K-Fold Cross-Validation."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }
    scores = cross_validate(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)
    return {
        'cv_accuracy_mean': float(np.mean(scores['test_accuracy'])),
        'cv_accuracy_std': float(np.std(scores['test_accuracy'])),
        'cv_precision_mean': float(np.mean(scores['test_precision'])),
        'cv_recall_mean': float(np.mean(scores['test_recall'])),
        'cv_f1_mean': float(np.mean(scores['test_f1'])),
        'cv_roc_auc_mean': float(np.mean(scores['test_roc_auc'])),
        'cv_roc_auc_std': float(np.std(scores['test_roc_auc']))
    }


def compute_test_metrics(model, X_test, y_test):
    """Computes comprehensive test set evaluation metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        'Accuracy': float(accuracy_score(y_test, y_pred)),
        'Precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'Recall': float(recall_score(y_test, y_pred)),
        'Specificity': float(specificity),
        'F1-Score': float(f1_score(y_test, y_pred)),
        'ROC-AUC': float(roc_auc_score(y_test, y_prob)),
        'Confusion Matrix': cm.tolist()
    }


def train_and_benchmark(output_dir=None, reports_dir=None):
    """
    Orchestrates full training, ensembling, and benchmarking.
    Saves trained models, metrics summary, and test set for reporting.
    """
    output_dir = str(output_dir or MODELS_DIR)
    reports_dir = str(reports_dir or REPORTS_DIR)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # 1. Run Data Preprocessing
    print(">>> 1. Loading and preprocessing data...")
    pipeline = DiabetesDataPipeline()
    data = pipeline.run_pipeline(artifacts_dir=output_dir)
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']

    # 2. Benchmark individual base classifiers via 5-Fold Cross-Validation
    print(">>> 2. Training base classifiers with 5-Fold Cross-Validation...")
    base_models = get_base_classifiers()
    model_records = {}
    fitted_models = {}
    auc_weights = []
    ensemble_estimators = []

    for name, model in base_models.items():
        print(f"  Training {name}...")
        cv_res = evaluate_model_cv(model, X_train, y_train, cv=5)
        # Fit on full training set
        model.fit(X_train, y_train)
        fitted_models[name] = model

        test_res = compute_test_metrics(model, X_test, y_test)
        record = {**cv_res, **test_res}
        model_records[name] = record

        # Record for weighted ensembling (weights = CV ROC-AUC)
        auc_weights.append(cv_res['cv_roc_auc_mean'])
        ensemble_estimators.append((name, model))

    # 3. Create Weighted Soft Voting Ensemble (IEEE Paper Proposed Model)
    print(">>> 3. Training Weighted Soft Voting Ensemble (IEEE Access methodology)...")
    weighted_ensemble = VotingClassifier(
        estimators=ensemble_estimators,
        voting='soft',
        weights=auc_weights
    )
    weighted_ensemble.fit(X_train, y_train)
    fitted_models['Weighted Soft Voting Ensemble (Proposed)'] = weighted_ensemble

    ens_cv = evaluate_model_cv(weighted_ensemble, X_train, y_train, cv=5)
    ens_test = compute_test_metrics(weighted_ensemble, X_test, y_test)
    model_records['Weighted Soft Voting Ensemble (Proposed)'] = {**ens_cv, **ens_test}

    # 4. Create Stacking Classifier (Meta-Learner: Logistic Regression)
    print(">>> 4. Training Stacking Classifier...")
    stacking_base = [
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=80, learning_rate=0.08, max_depth=3, random_state=42)),
        ('svm', SVC(C=1.0, kernel='rbf', probability=True, random_state=42)),
        ('lr', LogisticRegression(max_iter=1000, random_state=42))
    ]
    stacking_clf = StackingClassifier(
        estimators=stacking_base,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=5
    )
    stacking_clf.fit(X_train, y_train)
    fitted_models['Stacking Classifier'] = stacking_clf

    stk_cv = evaluate_model_cv(stacking_clf, X_train, y_train, cv=5)
    stk_test = compute_test_metrics(stacking_clf, X_test, y_test)
    model_records['Stacking Classifier'] = {**stk_cv, **stk_test}

    # 5. Format and Export Metrics Summary
    summary_rows = []
    for m_name, metrics in model_records.items():
        summary_rows.append({
            'Model': m_name,
            'CV Accuracy': f"{metrics['cv_accuracy_mean']:.4f} ± {metrics['cv_accuracy_std']:.4f}",
            'CV ROC-AUC': f"{metrics['cv_roc_auc_mean']:.4f} ± {metrics['cv_roc_auc_std']:.4f}",
            'Test Accuracy': round(metrics['Accuracy'], 4),
            'Test Precision': round(metrics['Precision'], 4),
            'Test Recall': round(metrics['Recall'], 4),
            'Test Specificity': round(metrics['Specificity'], 4),
            'Test F1-Score': round(metrics['F1-Score'], 4),
            'Test ROC-AUC': round(metrics['ROC-AUC'], 4)
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(by='Test ROC-AUC', ascending=False)
    summary_df.to_csv(os.path.join(reports_dir, 'metrics_summary.csv'), index=False)
    print("\n--- BENCHMARK RESULTS ---")
    print(summary_df.to_string(index=False))

    # Save models and test data for plotting and deployment
    joblib.dump(fitted_models, os.path.join(output_dir, 'trained_models.joblib'))
    joblib.dump({
        'X_test': X_test,
        'y_test': y_test,
        'X_train': X_train,
        'y_train': y_train,
        'feature_names': FEATURE_NAMES
    }, os.path.join(output_dir, 'processed_datasets.joblib'))

    with open(os.path.join(output_dir, 'model_metrics.json'), 'w') as f:
        json.dump(model_records, f, indent=4)

    print(f"\nAll models and summaries saved to '{output_dir}/' and '{reports_dir}/'")
    return fitted_models, summary_df


if __name__ == "__main__":
    train_and_benchmark()
