import os
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(BASE_DIR, "docs")

class TestPagesRouting(unittest.TestCase):
    def test_docs_directory_files_exist(self):
        required_files = [
            "index.html",
            "login.html",
            "style.css",
            "app.js",
            "models_data.js",
            ".nojekyll"
        ]
        for f in required_files:
            file_path = os.path.join(DOCS_DIR, f)
            self.assertTrue(os.path.isfile(file_path), f"Missing required file: {f}")

    def test_docs_docs_compatibility_redirects_exist(self):
        index_redir = os.path.join(DOCS_DIR, "docs", "index.html")
        login_redir = os.path.join(DOCS_DIR, "docs", "login.html")
        self.assertTrue(os.path.isfile(index_redir), "Missing docs/docs/index.html redirect")
        self.assertTrue(os.path.isfile(login_redir), "Missing docs/docs/login.html redirect")

        with open(index_redir, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("../", content)

        with open(login_redir, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("../login.html", content)

    def test_docs_index_no_forced_login_redirect(self):
        index_path = os.path.join(DOCS_DIR, "index.html")
        with open(index_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertNotIn("window.location.replace(\"login.html\")", content)
        self.assertIn("nav-login-btn", content)
        self.assertIn("nav-logout-btn", content)

    def test_readme_links(self):
        readme_path = os.path.join(BASE_DIR, "README.md")
        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("https://shabna-002.github.io/diabetes-prediction-using-machine-learning/", content)
        self.assertNotIn("diabetes-prediction-using-machine-learning/docs/)", content)

if __name__ == "__main__":
    unittest.main()
