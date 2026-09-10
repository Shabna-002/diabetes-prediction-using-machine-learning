"""
Evaluation and Visualization Module
Generates publication-quality charts:
1. Multi-model ROC Curves
2. Confusion Matrices Subplot Grid
3. Benchmark Performance Bar Charts
4. Feature Importance Analysis
5. Correlation Heatmap
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

from data_preprocessing import FEATURE_NAMES, ZERO_INVALID_COLS

try:
    from config import FIGURES_DIR, MODELS_DIR, REPORTS_DIR, RAW_DATASET_PATH
except ImportError:
    try:
        from src.config import FIGURES_DIR, MODELS_DIR, REPORTS_DIR, RAW_DATASET_PATH
    except ImportError:
        MODELS_DIR = "models"
        REPORTS_DIR = "reports"
        FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
        RAW_DATASET_PATH = "diabetes.csv"

# Configure styling for high quality publication-ready graphics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def plot_roc_curves(models, X_test, y_test, output_path="reports/figures/roc_curves.png"):
    """Plots and saves multi-model ROC curves with AUC annotations."""
    plt.figure(figsize=(10, 8), dpi=300)
    palette = sns.color_palette("tab10", len(models))

    for (name, model), color in zip(models.items(), palette):
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            is_proposed = "Proposed" in name
            linewidth = 3.0 if is_proposed else 1.8
            linestyle = '-' if not is_proposed else '--'
            plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})",
                     linewidth=linewidth, linestyle=linestyle, color=color)

    plt.plot([0, 1], [0, 1], color='#888888', linestyle=':', linewidth=1.5, label='Random Chance (AUC = 0.50)')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12, fontweight='bold')
    plt.title('Receiver Operating Characteristic (ROC) Curve Comparison', fontsize=14, fontweight='bold', pad=15)
    plt.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_confusion_matrices(models, X_test, y_test, output_path="reports/figures/confusion_matrices.png"):
    """Plots confusion matrix heatmaps for all models in a structured grid."""
    n_models = len(models)
    cols = 3
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4.2), dpi=300)
    axes = axes.flatten()

    for idx, (name, model) in enumerate(models.items()):
        ax = axes[idx]
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
            annot_kws={'size': 14, 'weight': 'bold'},
            xticklabels=['Non-Diabetic (0)', 'Diabetic (1)'],
            yticklabels=['Non-Diabetic (0)', 'Diabetic (1)']
        )
        ax.set_title(name, fontsize=11, fontweight='bold', pad=10)
        ax.set_xlabel('Predicted Label', fontsize=10)
        ax.set_ylabel('Actual Label', fontsize=10)

    # Hide extra unused subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle('Confusion Matrices on Test Set (N=154)', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_model_comparison_bar(summary_csv="reports/metrics_summary.csv", output_path="reports/figures/model_comparison.png"):
    """Generates a comprehensive comparative bar chart of model performance."""
    df = pd.read_csv(summary_csv)
    
    # Clean up model names for horizontal plot
    metrics = ['Test Accuracy', 'Test F1-Score', 'Test ROC-AUC']
    df_melted = df.melt(id_vars='Model', value_vars=metrics, var_name='Metric', value_name='Score')

    plt.figure(figsize=(12, 7), dpi=300)
    chart = sns.barplot(
        data=df_melted, x='Score', y='Model', hue='Metric',
        palette=['#2b5c8f', '#2a9d8f', '#e76f51']
    )
    plt.xlim(0.4, 1.0)
    plt.xlabel('Metric Score', fontsize=12, fontweight='bold')
    plt.ylabel('Model Architecture', fontsize=12, fontweight='bold')
    plt.title('Performance Comparison Across Machine Learning Classifiers', fontsize=14, fontweight='bold', pad=15)
    plt.legend(title='Metric', loc='lower right', frameon=True)
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_feature_importance(models, feature_names, output_path="reports/figures/feature_importance.png"):
    """Plots relative feature importances from Random Forest and Gradient Boosting."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # Random Forest Importances
    if 'Random Forest' in models:
        rf = models['Random Forest']
        rf_imp = pd.Series(rf.feature_importances_, index=feature_names).sort_values()
        rf_imp.plot(kind='barh', ax=ax1, color='#2b5c8f', edgecolor='black', alpha=0.85)
        ax1.set_title('Random Forest Feature Importance', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Gini Importance', fontsize=10)

    # Gradient Boosting Importances
    if 'Gradient Boosting' in models:
        gb = models['Gradient Boosting']
        gb_imp = pd.Series(gb.feature_importances_, index=feature_names).sort_values()
        gb_imp.plot(kind='barh', ax=ax2, color='#2a9d8f', edgecolor='black', alpha=0.85)
        ax2.set_title('Gradient Boosting Feature Importance', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Relative Importance', fontsize=10)

    plt.suptitle('Clinical Feature Importance for Diabetes Prediction', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_correlation_matrix(data_path="diabetes.csv", output_path="reports/figures/correlation_heatmap.png"):
    """Generates correlation heatmap of clinical features with Outcome."""
    df = pd.read_csv(data_path)
    # Handle zero replacements for realistic correlation
    for col in ZERO_INVALID_COLS:
        df[col] = df[col].replace(0, np.nan)

    corr = df.corr()

    plt.figure(figsize=(10, 8), dpi=300)
    sns.heatmap(
        corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1,
        square=True, linewidths=0.5, cbar_kws={'shrink': 0.8}
    )
    plt.title('Clinical Feature Correlation Matrix (Pima Indians Dataset)', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_all_visualizations(models_dir=None, reports_dir=None):
    """Generates all evaluation charts and saves them."""
    models_dir = str(models_dir or MODELS_DIR)
    reports_dir = str(reports_dir or REPORTS_DIR)
    figures_dir = os.path.join(reports_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    models = joblib.load(os.path.join(models_dir, 'trained_models.joblib'))
    data = joblib.load(os.path.join(models_dir, 'processed_datasets.joblib'))
    
    X_test, y_test = data['X_test'], data['y_test']
    feature_names = data['feature_names']

    print("Generating ROC Curves comparison...")
    plot_roc_curves(models, X_test, y_test, os.path.join(figures_dir, 'roc_curves.png'))

    print("Generating Confusion Matrices...")
    plot_confusion_matrices(models, X_test, y_test, os.path.join(figures_dir, 'confusion_matrices.png'))

    print("Generating Model Benchmark comparison chart...")
    plot_model_comparison_bar(os.path.join(reports_dir, 'metrics_summary.csv'), os.path.join(figures_dir, 'model_comparison.png'))

    print("Generating Feature Importance chart...")
    plot_feature_importance(models, feature_names, os.path.join(figures_dir, 'feature_importance.png'))

    print("Generating Correlation Heatmap...")
    plot_correlation_matrix(str(RAW_DATASET_PATH), os.path.join(figures_dir, 'correlation_heatmap.png'))

    print(f"\nAll figures generated successfully in '{figures_dir}/'!")


if __name__ == "__main__":
    generate_all_visualizations()
