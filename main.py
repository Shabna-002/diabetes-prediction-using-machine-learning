"""
Main Pipeline Driver for Diabetes Prediction System
Standard execution entry point for data preprocessing, model training, evaluation, inference, and dashboard launch.
"""

import os
import sys
import argparse
import subprocess
import pandas as pd

# Ensure src module is discoverable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from data_preprocessing import DiabetesDataPipeline
from model_training import train_and_benchmark
from evaluate_and_visualize import generate_all_visualizations
import predict


def run_preprocessing():
    print("\n" + "="*70)
    print(" [STEP 1/3] EXECUTING DATA PREPROCESSING & CLEANING PIPELINE")
    print("="*70)
    pipeline = DiabetesDataPipeline(data_path="diabetes.csv", test_size=0.2, random_state=42)
    data = pipeline.run_pipeline()
    
    # Also save preprocessed CSVs to data/processed/ for standard academic format
    os.makedirs(os.path.join("data", "processed"), exist_ok=True)
    train_df = data['X_train'].copy()
    train_df['Outcome'] = data['y_train'].values
    train_df.to_csv(os.path.join("data", "processed", "train_preprocessed.csv"), index=False)

    test_df = data['X_test'].copy()
    test_df['Outcome'] = data['y_test'].values
    test_df.to_csv(os.path.join("data", "processed", "test_preprocessed.csv"), index=False)

    print(f"-> Train Set: {data['X_train'].shape[0]} samples, 8 features")
    print(f"-> Test Set:  {data['X_test'].shape[0]} samples, 8 features")
    print("-> Preprocessed datasets saved to 'data/processed/' and 'models/' successfully.\n")
    return data


def run_training():
    print("\n" + "="*70)
    print(" [STEP 2/3] TRAINING & TUNING MACHINE LEARNING CLASSIFIERS")
    print("="*70)
    fitted_models, summary_df = train_and_benchmark()
    print("-> Models serialized to 'models/trained_models.joblib'.\n")
    return fitted_models, summary_df


def run_evaluation():
    print("\n" + "="*70)
    print(" [STEP 3/3] EVALUATION & GENERATING FIGURES / REPORTS")
    print("="*70)
    generate_all_visualizations()
    summary_path = os.path.join("reports", "metrics_summary.csv")
    if os.path.exists(summary_path):
        summary_df = pd.read_csv(summary_path)
        print("\nBenchmarking Summary Table:")
        print(summary_df.to_string(index=False))
    print("\n-> Visualizations saved to 'reports/figures/':")
    print("   • correlation_heatmap.png")
    print("   • confusion_matrices.png")
    print("   • roc_curves.png")
    print("   • feature_importance.png")
    print("   • model_comparison.png")
    print("-> Summary metrics saved to 'reports/metrics_summary.csv'.\n")


def run_inference_demo():
    print("\n" + "="*70)
    print(" [DEMO] RUNNING SAMPLE PATIENT PREDICTIONS")
    print("="*70)
    predict.main([])


def launch_app():
    print("\n" + "="*70)
    print(" LAUNCHING STREAMLIT CLINICAL DECISION SUPPORT DASHBOARD")
    print("="*70)
    print("Press Ctrl+C to terminate the web server when done.\n")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])


def main():
    parser = argparse.ArgumentParser(description="Unified Driver for Diabetes Prediction ML System")
    parser.add_argument(
        '--mode',
        type=str,
        default='all',
        choices=['all', 'preprocess', 'train', 'evaluate', 'predict', 'app'],
        help="Pipeline stage to execute: 'preprocess', 'train', 'evaluate', 'predict', 'app', or 'all'"
    )

    args = parser.parse_args()

    if args.mode == 'preprocess':
        run_preprocessing()
    elif args.mode == 'train':
        run_training()
    elif args.mode == 'evaluate':
        run_evaluation()
    elif args.mode == 'predict':
        run_inference_demo()
    elif args.mode == 'app':
        launch_app()
    elif args.mode == 'all':
        run_preprocessing()
        run_training()
        run_evaluation()
        run_inference_demo()
        print("="*70)
        print(" ALL PIPELINE STAGES COMPLETED SUCCESSFULLY!")
        print(" To start the interactive web application, run:")
        print("   py -3.10 main.py --mode app  (or double click run_app.bat)")
        print("="*70 + "\n")


if __name__ == "__main__":
    main()
