import os
import unittest
import numpy as np
import pandas as pd
import joblib

# Ensure SQLite fallback is used during test to avoid needing local MySQL server running
os.environ["USE_SQLITE_ONLY"] = "1"

from app import app, FEATURES, ZERO_AS_MISSING, get_model, MODEL_CHOICES


class DiabetesMLFullTestSuite(unittest.TestCase):
    def setUp(self):
        self.app = app
        self.app.config["TESTING"] = True
        self.app.config["WTF_CSRF_ENABLED"] = False
        self.client = self.app.test_client()

    def test_01_all_models_exist_and_load(self):
        """Verify that all 6 required classifier pipelines exist in model/ and load properly."""
        required_models = [
            "best", "logistic_regression", "decision_tree",
            "random_forest", "knn", "svm", "naive_bayes"
        ]
        for key in required_models:
            clf, label = get_model(key)
            self.assertIsNotNone(clf, f"Model {key} failed to load from model directory")
            self.assertIsNotNone(label, f"Label for {key} is None")
            self.assertTrue(hasattr(clf, "predict"), f"Loaded object for {key} lacks predict method")

    def test_02_leak_free_pipeline_inference(self):
        """Verify that the saved pipeline can handle raw clinical inputs with zeros and predict cleanly."""
        clf, _ = get_model("best")
        
        # Test sample with 0 in Insulin and SkinThickness (must be imputed without error)
        sample_input = {
            "Pregnancies": 2.0,
            "Glucose": 130.0,
            "BloodPressure": 70.0,
            "SkinThickness": 0.0,  # Biologically missing
            "Insulin": 0.0,        # Biologically missing
            "BMI": 28.4,
            "DiabetesPedigreeFunction": 0.35,
            "Age": 32.0,
            "Gender": 0.0,
            "HbA1c": 6.2,
            "PhysicalActivity": 1.0,
            "SmokingStatus": 0.0
        }
        
        # In app inference, zeros in ZERO_AS_MISSING are converted to np.nan
        for col in ZERO_AS_MISSING:
            if sample_input[col] == 0:
                sample_input[col] = np.nan
                
        df = pd.DataFrame([sample_input], columns=FEATURES)
        pred = clf.predict(df)
        self.assertIn(int(pred[0]), [0, 1])
        
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(df)
            self.assertEqual(proba.shape, (1, 2))
            self.assertAlmostEqual(proba[0].sum(), 1.0, places=4)

    def test_03_home_route(self):
        """Verify home page loads and displays benchmark info."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Dia", response.data)
        self.assertIn(b"Prediction", response.data)

    def test_04_models_benchmark_route(self):
        """Verify the /models benchmark comparison page loads with charts and all 6 classifiers."""
        response = self.client.get("/models")
        self.assertEqual(response.status_code, 200)
        content = response.data.decode("utf-8")
        
        # Check all 6 required models are listed
        self.assertIn("Logistic Regression", content)
        self.assertIn("Decision Tree", content)
        self.assertIn("Random Forest", content)
        self.assertIn("K-Nearest Neighbours", content)
        self.assertIn("Support Vector Machine", content)
        self.assertIn("Naive Bayes", content)
        
        # Check chart canvas elements are present
        self.assertIn("metricsComparisonChart", content)
        self.assertIn("rocCurvesChart", content)
        
        # Check confusion matrix and CV F2 mentions
        self.assertIn("CV F2", content)
        self.assertIn("Confusion Matrix", content)

    def test_05_predict_get_route(self):
        """Verify GET /predict displays form and all 6 candidate algorithms."""
        response = self.client.get("/predict")
        self.assertEqual(response.status_code, 200)
        content = response.data.decode("utf-8")
        self.assertIn("Patient Risk Assessment", content)
        self.assertIn("naive_bayes", content)
        self.assertIn("logistic_regression", content)

    def test_06_predict_post_route_all_classifiers(self):
        """Verify POST /predict succeeds and returns risk probability for all classifiers."""
        test_models = ["best", "naive_bayes", "logistic_regression", "random_forest", "knn", "svm", "decision_tree"]
        
        for m in test_models:
            payload = {
                "model_choice": m,
                "patient_name": f"Test Patient {m}",
                "pregnancies": "2",
                "glucose": "140",
                "blood_pressure": "72",
                "skin_thickness": "0",  # Zero handled
                "insulin": "0",         # Zero handled
                "bmi": "31.2",
                "diabetes_pedigree": "0.45",
                "age": "45",
                "gender": "0",
                "hba1c": "6.8",
                "physical_activity": "1",
                "smoking_status": "0"
            }
            response = self.client.post("/predict", data=payload, follow_redirects=True)
            self.assertEqual(response.status_code, 200)
            content = response.data.decode("utf-8")
            self.assertIn("Estimated Risk Probability", content)
            self.assertIn("Test Patient", content)

    def test_07_dashboard_and_history_routes(self):
        """Verify dashboard and history reflect recorded predictions."""
        resp_dash = self.client.get("/dashboard")
        self.assertEqual(resp_dash.status_code, 200)
        self.assertIn(b"Prediction Dashboard", resp_dash.data)

        resp_hist = self.client.get("/history")
        self.assertEqual(resp_hist.status_code, 200)
        self.assertIn(b"Prediction History", resp_hist.data)

    def test_08_login_and_authentication(self):
        """Verify login functionality with demo credentials and session management."""
        # 1. Login page GET
        resp = self.client.get("/login")
        self.assertEqual(resp.status_code, 200)
        
        # 2. Login with valid demo credentials
        login_data = {"username": "admin", "password": "admin123"}
        resp = self.client.post("/login", data=login_data, follow_redirects=True)
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Welcome", resp.data)
        
        # 3. Logout
        resp_logout = self.client.get("/logout", follow_redirects=True)
        self.assertEqual(resp_logout.status_code, 200)
        self.assertIn(b"signed out", resp_logout.data)


if __name__ == "__main__":
    unittest.main()
