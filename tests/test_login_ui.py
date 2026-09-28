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
            "remember_me"
        ]
        
        for item in required_elements:
            self.assertIn(item, html, f"Missing required element: {item}")

        # Explicitly verify removed elements are NOT present
        prohibited_elements = [
            "Forgot Password?",
            "Register Now",
            "Don't have an account yet?",
            "Return to Home",
            "Register Account"
        ]
        for item in prohibited_elements:
            self.assertNotIn(item, html, f"Prohibited element found: {item}")

    def test_static_login_files_removed_elements(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for path in [os.path.join(base_dir, "docs", "login.html"), os.path.join(base_dir, "login.html")]:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertNotIn("Forgot Password?", content, f"Forgot Password found in {path}")
            self.assertNotIn("Register Now", content, f"Register Now found in {path}")
            self.assertNotIn("Don't have an account yet?", content, f"Don't have an account yet found in {path}")
            self.assertNotIn("Return to Home", content, f"Return to Home found in {path}")
            self.assertNotIn("Register Account", content, f"Register Account found in {path}")

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
