import unittest
import os
os.environ["USE_SQLITE_ONLY"] = "1"
from app import app


class TestLoginUI(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_login_page_elements(self):
        res = self.client.get("/login")
        self.assertEqual(res.status_code, 200)
        html = res.data.decode("utf-8")
        
        required_elements = [
            "Welcome to <span class=\"bg-gradient-to-r",
            "DiabetesPredict AI",
            "Smarter Predictions. Healthier Tomorrow.",
            "AI-Powered Prediction",
            "Early Risk Insights",
            "Secure Access",
            "Welcome Back",
            "Login to your account",
            "Digital Glucometer",
            "104",
            "mg/dL",
            "ecg-line",
            "name=\"username\"",
            "name=\"password\"",
            "togglePasswordVisibility",
            "remember_me",
            "Forgot Password?",
            "Register Now"
        ]
        
        for item in required_elements:
            self.assertIn(item, html, f"Missing element: {item}")

    def test_login_post_flow(self):
        # Valid login
        res = self.client.post("/login", data={"username": "admin", "password": "admin123"}, follow_redirects=False)
        self.assertEqual(res.status_code, 302)
        
        # Log out so session is cleared
        self.client.get("/logout")
        
        # Invalid login
        res_fail = self.client.post("/login", data={"username": "admin", "password": "wrongpassword"}, follow_redirects=True)
        self.assertEqual(res_fail.status_code, 200)
        self.assertIn("Invalid username or password", res_fail.data.decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
