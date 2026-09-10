"""
Backend Web Server for Diabetes Prediction System
Serves the web portal frontend and provides real-time REST API endpoints
for machine learning inference, model metrics, and batch screening.
Zero external server dependencies required (uses Python standard library http.server).
"""

import os
import sys
import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse
import pandas as pd
import numpy as np

# Ensure project modules are discoverable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

from predict import load_artifacts, preprocess_patient
from data_preprocessing import FEATURE_NAMES

PORT = int(os.environ.get("PORT", 5000))
WEB_DIR = os.path.join(BASE_DIR, 'website')

# Cache model artifacts at startup
print("[INFO] Loading ML artifacts...")
try:
    IMPUTER, SCALER, IQR_BOUNDS, MODELS = load_artifacts(os.path.join(BASE_DIR, "models"))
    print(f"[INFO] Successfully loaded {len(MODELS)} trained ML models.")
except Exception as e:
    print(f"[ERROR] Failed to load models: {e}")
    IMPUTER, SCALER, IQR_BOUNDS, MODELS = None, None, None, {}

# Load metrics summary
METRICS_PATH = os.path.join(BASE_DIR, "reports", "metrics_summary.csv")
if os.path.exists(METRICS_PATH):
    METRICS_DF = pd.read_csv(METRICS_PATH)
else:
    METRICS_DF = pd.DataFrame()


def evaluate_clinical_biomarkers(data):
    """Evaluates each biomarker against clinical reference ranges."""
    evaluations = []
    
    # Glucose
    glucose = data.get('Glucose', 0)
    if glucose < 100:
        evaluations.append({"name": "Fasting Plasma Glucose", "value": f"{glucose} mg/dL", "status": "Normal", "badge": "success", "note": "Optimal glycemic level (<100 mg/dL)"})
    elif glucose <= 125:
        evaluations.append({"name": "Fasting Plasma Glucose", "value": f"{glucose} mg/dL", "status": "Pre-diabetic", "badge": "warning", "note": "Impaired fasting glucose (100-125 mg/dL)"})
    else:
        evaluations.append({"name": "Fasting Plasma Glucose", "value": f"{glucose} mg/dL", "status": "Elevated", "badge": "danger", "note": "Diabetic threshold exceeded (>=126 mg/dL)"})

    # BMI
    bmi = data.get('BMI', 0)
    if bmi < 18.5:
        evaluations.append({"name": "Body Mass Index (BMI)", "value": f"{bmi:.1f} kg/m²", "status": "Underweight", "badge": "info", "note": "BMI is below normal range (<18.5)"})
    elif bmi < 25.0:
        evaluations.append({"name": "Body Mass Index (BMI)", "value": f"{bmi:.1f} kg/m²", "status": "Normal", "badge": "success", "note": "Healthy weight range (18.5-24.9)"})
    elif bmi < 30.0:
        evaluations.append({"name": "Body Mass Index (BMI)", "value": f"{bmi:.1f} kg/m²", "status": "Overweight", "badge": "warning", "note": "Pre-obesity range (25.0-29.9)"})
    else:
        evaluations.append({"name": "Body Mass Index (BMI)", "value": f"{bmi:.1f} kg/m²", "status": "Obese", "badge": "danger", "note": "Class I+ obesity indicator (>=30.0)"})

    # Blood Pressure
    bp = data.get('BloodPressure', 0)
    if bp < 80:
        evaluations.append({"name": "Diastolic Blood Pressure", "value": f"{bp} mm Hg", "status": "Normal", "badge": "success", "note": "Diastolic within normal limits (<80 mm Hg)"})
    elif bp <= 89:
        evaluations.append({"name": "Diastolic Blood Pressure", "value": f"{bp} mm Hg", "status": "Pre-hypertension", "badge": "warning", "note": "Stage 1 elevated diastolic (80-89 mm Hg)"})
    else:
        evaluations.append({"name": "Diastolic Blood Pressure", "value": f"{bp} mm Hg", "status": "Hypertensive", "badge": "danger", "note": "Hypertensive crisis / Stage 2 (>=90 mm Hg)"})

    # Insulin
    insulin = data.get('Insulin', 0)
    if insulin < 16:
        evaluations.append({"name": "2-Hour Serum Insulin", "value": f"{insulin} uU/mL", "status": "Low / Normal", "badge": "info", "note": "Lower end of normal fasting serum insulin"})
    elif insulin <= 166:
        evaluations.append({"name": "2-Hour Serum Insulin", "value": f"{insulin} uU/mL", "status": "Normal", "badge": "success", "note": "Physiologic insulin range (16-166 uU/mL)"})
    else:
        evaluations.append({"name": "2-Hour Serum Insulin", "value": f"{insulin} uU/mL", "status": "Hyperinsulinemia", "badge": "danger", "note": "Elevated insulin indicates insulin resistance"})

    # Age & DPF
    dpf = data.get('DiabetesPedigreeFunction', 0)
    if dpf >= 0.6:
        evaluations.append({"name": "Diabetes Pedigree Score", "value": f"{dpf:.2f}", "status": "High Genetic Risk", "badge": "warning", "note": "Substantial hereditary family risk factor"})
    else:
        evaluations.append({"name": "Diabetes Pedigree Score", "value": f"{dpf:.2f}", "status": "Normal / Moderate", "badge": "success", "note": "Moderate to low hereditary score"})

    return evaluations


class DiabetesRequestHandler(SimpleHTTPRequestHandler):
    """Custom request handler with REST API endpoints."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/models":
            self.send_json_response(self.get_models_list())
        elif path == "/api/metrics":
            self.send_json_response(self.get_metrics_data())
        else:
            super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/predict":
            try:
                length = int(self.headers.get('Content-Length', 0))
                raw_data = self.rfile.read(length)
                payload = json.loads(raw_data.decode('utf-8'))
                result = self.perform_prediction(payload)
                self.send_json_response(result)
            except Exception as e:
                self.send_json_response({"error": str(e)}, status=400)
        elif path == "/api/batch":
            try:
                length = int(self.headers.get('Content-Length', 0))
                raw_data = self.rfile.read(length)
                payload = json.loads(raw_data.decode('utf-8'))
                result = self.perform_batch_prediction(payload)
                self.send_json_response(result)
            except Exception as e:
                self.send_json_response({"error": str(e)}, status=400)
        else:
            self.send_json_response({"error": "Endpoint not found"}, status=404)

    def send_json_response(self, data, status=200):
        body = json.dumps(data).encode('utf-8')
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def get_models_list(self):
        model_list = []
        for name in MODELS.keys():
            row = METRICS_DF[METRICS_DF['Model'] == name] if not METRICS_DF.empty else None
            acc = float(row['Test Accuracy'].values[0]) if row is not None and not row.empty else 0.75
            auc = float(row['Test ROC-AUC'].values[0]) if row is not None and not row.empty else 0.80
            model_list.append({
                "name": name,
                "accuracy": round(acc * 100, 1),
                "roc_auc": round(auc, 3),
                "is_default": "Ensemble" in name
            })
        return {"models": model_list}

    def get_metrics_data(self):
        if METRICS_DF.empty:
            return {"metrics": []}
        return {"metrics": METRICS_DF.to_dict(orient="records")}

    def perform_prediction(self, payload):
        patient_dict = {
            'Pregnancies': float(payload.get('Pregnancies', 1)),
            'Glucose': float(payload.get('Glucose', 120)),
            'BloodPressure': float(payload.get('BloodPressure', 70)),
            'SkinThickness': float(payload.get('SkinThickness', 20)),
            'Insulin': float(payload.get('Insulin', 80)),
            'BMI': float(payload.get('BMI', 25.0)),
            'DiabetesPedigreeFunction': float(payload.get('DiabetesPedigreeFunction', 0.45)),
            'Age': float(payload.get('Age', 33))
        }

        model_name = payload.get('model_name', 'Weighted Soft Voting Ensemble (Proposed)')
        if model_name not in MODELS:
            model_name = list(MODELS.keys())[0] if MODELS else None

        if not model_name or model_name not in MODELS:
            raise ValueError("No valid ML model available.")

        model = MODELS[model_name]
        X_proc = preprocess_patient(patient_dict, IMPUTER, SCALER, IQR_BOUNDS)

        pred_code = int(model.predict(X_proc)[0])
        pred_prob = float(model.predict_proba(X_proc)[0, 1])

        if pred_prob < 0.35:
            risk_tier = "Low Risk"
            risk_class = "low"
            color = "#10B981"
            recommendation = "Maintain balanced nutrition, regular physical activity, and routine annual health checkups."
        elif pred_prob < 0.65:
            risk_tier = "Moderate Risk"
            risk_class = "moderate"
            color = "#F59E0B"
            recommendation = "Borderline glucose / metabolic markers detected. Dietary modification and medical consultation recommended."
        else:
            risk_tier = "High Risk"
            risk_class = "high"
            color = "#EF4444"
            recommendation = "Significant risk indicators detected. Immediate consultation with an endocrinologist for HbA1c and oral glucose tolerance testing is strongly advised."

        biomarkers = evaluate_clinical_biomarkers(patient_dict)

        # Comparative predictions from other primary models
        multi_model_results = {}
        primary_models = ["Logistic Regression", "Decision Tree", "Random Forest", "Support Vector Machine", "Weighted Soft Voting Ensemble (Proposed)"]
        for m_name in primary_models:
            if m_name in MODELS:
                m_obj = MODELS[m_name]
                m_prob = float(m_obj.predict_proba(X_proc)[0, 1])
                m_code = int(m_obj.predict(X_proc)[0])
                multi_model_results[m_name] = {
                    "prediction": "Diabetic" if m_code == 1 else "Non-Diabetic",
                    "probability": round(m_prob, 4),
                    "probability_percent": f"{m_prob * 100:.1f}%"
                }

        return {
            "prediction": "Diabetic" if pred_code == 1 else "Non-Diabetic",
            "prediction_code": pred_code,
            "probability": round(pred_prob, 4),
            "probability_percent": f"{pred_prob * 100:.1f}%",
            "risk_tier": risk_tier,
            "risk_class": risk_class,
            "color": color,
            "recommendation": recommendation,
            "model_used": model_name,
            "biomarkers": biomarkers,
            "patient_data": patient_dict,
            "multi_model_comparison": multi_model_results
        }

    def perform_batch_prediction(self, payload):
        patients = payload.get('patients', [])
        model_name = payload.get('model_name', 'Weighted Soft Voting Ensemble (Proposed)')
        if model_name not in MODELS:
            model_name = list(MODELS.keys())[0]

        model = MODELS[model_name]
        results = []

        for p in patients:
            p_clean = {k: float(p.get(k, 0)) for k in FEATURE_NAMES}
            X_proc = preprocess_patient(p_clean, IMPUTER, SCALER, IQR_BOUNDS)
            pred_code = int(model.predict(X_proc)[0])
            pred_prob = float(model.predict_proba(X_proc)[0, 1])

            risk_tier = "Low" if pred_prob < 0.35 else ("Moderate" if pred_prob < 0.65 else "High")
            results.append({
                **p_clean,
                "Prediction": "Diabetic" if pred_code == 1 else "Non-Diabetic",
                "Probability": f"{pred_prob * 100:.1f}%",
                "Risk": risk_tier
            })

        return {"results": results, "total": len(results)}


try:
    from http.server import ThreadingHTTPServer as DefaultHTTPServer
except ImportError:
    from http.server import HTTPServer as DefaultHTTPServer


def run_server():
    os.makedirs(WEB_DIR, exist_ok=True)
    server_address = ('0.0.0.0', PORT)
    httpd = DefaultHTTPServer(server_address, DiabetesRequestHandler)
    print("=" * 65)
    print(f" Diabetes ML Clinical Decision Portal Running Successfully!")
    print(f" Web Portal URL: http://localhost:{PORT}")
    print(f" REST API:       http://localhost:{PORT}/api/predict")
    print(f" Press Ctrl+C in terminal to stop server.")
    print("=" * 65)
    sys.stdout.flush()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Server stopped gracefully.")
        httpd.server_close()


if __name__ == '__main__':
    run_server()
