import os
import json
import sqlite3
from datetime import datetime

import joblib
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash

try:
    import mysql.connector
    MYSQL_LIBRARY_AVAILABLE = True
except ImportError:
    MYSQL_LIBRARY_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
SQLITE_DB_PATH = os.path.join(BASE_DIR, "data", "predictions.db")
METRICS_PATH = os.path.join(MODEL_DIR, "model_metrics.json")

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-this-secret-key")

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "127.0.0.1"),
    "user": os.environ.get("DB_USER", "root"),
    "password": os.environ.get("DB_PASSWORD", ""),
    "database": os.environ.get("DB_NAME", "diabetes_prediction_db"),
}

MODEL_CHOICES = {
    "best": "Best Performing Model",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "svm": "Support Vector Machine (SVM)",
    "logistic_regression": "Logistic Regression"
}

# In-memory model cache
loaded_models = {}

def get_metrics_data():
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def get_model(model_key="best"):
    metrics = get_metrics_data()
    best_name = metrics.get("best_model", "Decision Tree") if metrics else "Decision Tree"
    
    if model_key in loaded_models:
        return loaded_models[model_key], MODEL_CHOICES.get(model_key, model_key)
    
    filename_map = {
        "best": "diabetes_model.joblib",
        "decision_tree": "decision_tree.joblib",
        "random_forest": "random_forest.joblib",
        "svm": "svm.joblib",
        "logistic_regression": "logistic_regression.joblib"
    }
    
    fname = filename_map.get(model_key, "diabetes_model.joblib")
    path = os.path.join(MODEL_DIR, fname)
    
    # Fallback to diabetes_model.joblib if specific one is absent
    if not os.path.exists(path):
        path = os.path.join(MODEL_DIR, "diabetes_model.joblib")
        
    if os.path.exists(path):
        clf = joblib.load(path)
        loaded_models[model_key] = clf
        label = f"Best Model ({best_name})" if model_key == "best" else MODEL_CHOICES.get(model_key, model_key)
        return clf, label
    
    return None, None

def init_sqlite_db():
    os.makedirs(os.path.dirname(SQLITE_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT,
            password_hash TEXT NOT NULL,
            full_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT,
            model_used TEXT,
            pregnancies REAL,
            glucose REAL,
            blood_pressure REAL,
            skin_thickness REAL,
            insulin REAL,
            bmi REAL,
            diabetes_pedigree REAL,
            age REAL,
            prediction INTEGER,
            probability REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Seed default accounts if users table is empty
    cur.execute("SELECT COUNT(*) FROM users")
    if cur.fetchone()[0] == 0:
        admin_hash = generate_password_hash("admin123")
        doc_hash = generate_password_hash("doctor123")
        cur.execute(
            "INSERT INTO users (username, email, password_hash, full_name) VALUES (?, ?, ?, ?)",
            ("admin", "admin@hospital.org", admin_hash, "Administrator")
        )
        cur.execute(
            "INSERT INTO users (username, email, password_hash, full_name) VALUES (?, ?, ?, ?)",
            ("doctor", "doctor@hospital.org", doc_hash, "Dr. Clinician")
        )
    conn.commit()
    conn.close()

init_sqlite_db()

def find_user(username_or_email):
    q_mysql = "SELECT * FROM users WHERE username = %s OR email = %s LIMIT 1"
    q_sqlite = "SELECT * FROM users WHERE username = ? OR email = ? LIMIT 1"
    rows, _ = db_execute(q_mysql, q_sqlite, (username_or_email, username_or_email), fetch_mode="all_dict")
    return rows[0] if rows else None

def register_user(username, email, password, full_name=""):
    pw_hash = generate_password_hash(password)
    q_mysql = "INSERT INTO users (username, email, password_hash, full_name) VALUES (%s, %s, %s, %s)"
    q_sqlite = "INSERT INTO users (username, email, password_hash, full_name) VALUES (?, ?, ?, ?)"
    db_execute(q_mysql, q_sqlite, (username, email, pw_hash, full_name))

def db_execute(query_mysql, query_sqlite, params=(), fetch_mode=None):
    """
    Executes a query against MySQL if accessible; otherwise falls back gracefully to SQLite.
    fetch_mode: None (insert/update/delete), 'one', 'all_dict'
    """
    # 1. Try MySQL if enabled
    if MYSQL_LIBRARY_AVAILABLE and os.environ.get("USE_SQLITE_ONLY", "").lower() not in ("1", "true"):
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            cur = conn.cursor(dictionary=True if fetch_mode == "all_dict" else False)
            cur.execute(query_mysql, params)
            
            result = None
            if fetch_mode == "one":
                row = cur.fetchone()
                result = row[0] if row else 0
            elif fetch_mode == "all_dict":
                result = cur.fetchall()
            else:
                conn.commit()
                
            cur.close()
            conn.close()
            return result, "MySQL"
        except Exception:
            # MySQL connection failed, fallback to SQLite below
            pass

    # 2. SQLite fallback
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row if fetch_mode == "all_dict" else None
    cur = conn.cursor()
    cur.execute(query_sqlite, params)
    
    result = None
    if fetch_mode == "one":
        row = cur.fetchone()
        result = row[0] if row else 0
    elif fetch_mode == "all_dict":
        rows = cur.fetchall()
        result = [dict(r) for r in rows]
    else:
        conn.commit()
        
    cur.close()
    conn.close()
    return result, "SQLite"

@app.route("/")
def home():
    metrics = get_metrics_data()
    best_model = metrics.get("best_model", "Decision Tree") if metrics else "Decision Tree"
    return render_template("index.html", best_model=best_model)

@app.route("/models")
def models():
    metrics = get_metrics_data()
    return render_template("models.html", metrics=metrics)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        metrics = get_metrics_data()
        best_name = metrics.get("best_model", "Decision Tree") if metrics else "Decision Tree"
        return render_template("predict.html", best_name=best_name, choices=MODEL_CHOICES)

    try:
        model_key = request.form.get("model_choice", "best")
        patient_name = request.form.get("patient_name", "").strip() or None
        
        values = {
            "Pregnancies": float(request.form["pregnancies"]),
            "Glucose": float(request.form["glucose"]),
            "BloodPressure": float(request.form["blood_pressure"]),
            "SkinThickness": float(request.form["skin_thickness"]),
            "Insulin": float(request.form["insulin"]),
            "BMI": float(request.form["bmi"]),
            "DiabetesPedigreeFunction": float(request.form["diabetes_pedigree"]),
            "Age": float(request.form["age"]),
        }

        clf, model_label = get_model(model_key)
        if clf is None:
            flash("Trained model not found. Please run train_model.py first.", "error")
            return redirect(url_for("predict"))

        X = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)
        prediction = int(clf.predict(X)[0])
        probability = float(clf.predict_proba(X)[0][1]) if hasattr(clf, "predict_proba") else float(prediction)

        # Insert record into database
        q_mysql = """
            INSERT INTO predictions
            (patient_name, model_used, pregnancies, glucose, blood_pressure, skin_thickness,
             insulin, bmi, diabetes_pedigree, age, prediction, probability)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """
        q_sqlite = """
            INSERT INTO predictions
            (patient_name, model_used, pregnancies, glucose, blood_pressure, skin_thickness,
             insulin, bmi, diabetes_pedigree, age, prediction, probability)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """
        params = (
            patient_name, model_label, values["Pregnancies"], values["Glucose"],
            values["BloodPressure"], values["SkinThickness"], values["Insulin"],
            values["BMI"], values["DiabetesPedigreeFunction"], values["Age"],
            prediction, probability
        )
        
        try:
            _, db_type = db_execute(q_mysql, q_sqlite, params)
        except Exception:
            # Table in MySQL might not have model_used yet, try fallback without model_used
            try:
                db_execute(
                    """INSERT INTO predictions (patient_name, pregnancies, glucose, blood_pressure, skin_thickness,
                                                insulin, bmi, diabetes_pedigree, age, prediction, probability)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    q_sqlite,
                    (patient_name, values["Pregnancies"], values["Glucose"], values["BloodPressure"],
                     values["SkinThickness"], values["Insulin"], values["BMI"],
                     values["DiabetesPedigreeFunction"], values["Age"], prediction, probability)
                )
            except Exception as e:
                flash(f"Warning: Database storage issue: {e}", "warning")

        result = "Higher predicted risk" if prediction == 1 else "Lower predicted risk"
        return render_template(
            "result.html",
            result=result,
            prediction=prediction,
            probability=round(probability * 100, 2),
            patient_name=patient_name,
            model_used=model_label,
            inputs=values
        )

    except Exception as exc:
        flash(f"Could not complete prediction: {exc}", "error")
        return redirect(url_for("predict"))

@app.route("/history")
def history():
    try:
        q_mysql = "SELECT * FROM predictions ORDER BY created_at DESC LIMIT 100"
        q_sqlite = "SELECT * FROM predictions ORDER BY created_at DESC LIMIT 100"
        rows, db_type = db_execute(q_mysql, q_sqlite, fetch_mode="all_dict")
        return render_template("history.html", rows=rows, db_type=db_type)
    except Exception as exc:
        flash(f"Database query error: {exc}", "error")
        return redirect(url_for("home"))

@app.route("/dashboard")
def dashboard():
    try:
        total, db_type = db_execute("SELECT COUNT(*) FROM predictions", "SELECT COUNT(*) FROM predictions", fetch_mode="one")
        positive, _ = db_execute("SELECT COUNT(*) FROM predictions WHERE prediction = 1", "SELECT COUNT(*) FROM predictions WHERE prediction = 1", fetch_mode="one")
        negative, _ = db_execute("SELECT COUNT(*) FROM predictions WHERE prediction = 0", "SELECT COUNT(*) FROM predictions WHERE prediction = 0", fetch_mode="one")
        return render_template("dashboard.html", total=total, positive=positive, negative=negative, db_type=db_type)
    except Exception as exc:
        flash(f"Database query error: {exc}", "error")
        return redirect(url_for("home"))

@app.route("/clear-history", methods=["POST"])
def clear_history():
    try:
        db_execute("DELETE FROM predictions", "DELETE FROM predictions")
        flash("Prediction history cleared successfully.", "success")
    except Exception as exc:
        flash(f"Database error: {exc}", "error")
@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user"):
        return redirect(url_for("home"))
        
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        
        if not username or not password:
            flash("Please enter both username and password.", "error")
            return render_template("login.html")
            
        user = find_user(username)
        if user and check_password_hash(user["password_hash"], password):
            session["user"] = user["username"]
            session["user_id"] = user["id"]
            session["full_name"] = user.get("full_name") or user["username"]
            flash(f"Welcome, {session['full_name']}!", "success")
            next_url = request.args.get("next") or url_for("home")
            return redirect(next_url)
        else:
            flash("Invalid username or password. You can try the demo account: admin / admin123", "error")
            
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if session.get("user"):
        return redirect(url_for("home"))
        
    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()
        confirm_password = request.form.get("confirm_password", "").strip()
        
        if not username or not password:
            flash("Username and password are required.", "error")
            return render_template("register.html")
            
        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return render_template("register.html")
            
        if len(password) < 6:
            flash("Password must be at least 6 characters long.", "error")
            return render_template("register.html")
            
        existing = find_user(username)
        if existing:
            flash("Username already taken. Please select a different username.", "error")
            return render_template("register.html")
            
        try:
            register_user(username, email, password, full_name)
            flash("Registration successful! Please log in with your new credentials.", "success")
            return redirect(url_for("login"))
        except Exception as exc:
            flash(f"Error during registration: {exc}", "error")
            
    return render_template("register.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("You have been signed out.", "success")
    return redirect(url_for("home"))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

