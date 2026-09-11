"""
Streamlit Web Application: Diabetes Prediction & Clinical Decision Support System
Integrates real-time patient assessment, multi-patient screening, exploratory data analysis (EDA),
and multi-model benchmarking based on the IEEE Access ensemble methodology.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from data_preprocessing import FEATURE_NAMES, ZERO_INVALID_COLS
from predict import load_artifacts, preprocess_patient

# Page configuration
st.set_page_config(
    page_title="Diabetes ML Clinical Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .card-diabetic {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        border: 2px solid #EF4444;
        border-radius: 12px;
        padding: 20px;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    .card-nondiabetic {
        background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
        border: 2px solid #22C55E;
        border-radius: 12px;
        padding: 20px;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    .metric-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .badge-high { background-color: #FEE2E2; color: #DC2626; border: 1px solid #DC2626; }
    .badge-moderate { background-color: #FEF3C7; color: #D97706; border: 1px solid #D97706; }
    .badge-low { background-color: #DCFCE7; color: #16A34A; border: 1px solid #16A34A; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        border-radius: 6px 6px 0px 0px;
        padding: 10px 18px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_cached_artifacts():
    """Loads and caches trained models and preprocessing pipelines."""
    imputer, scaler, iqr_bounds, models = load_artifacts("models")
    metrics_df = pd.read_csv("reports/metrics_summary.csv")
    raw_data = pd.read_csv("diabetes.csv")
    return imputer, scaler, iqr_bounds, models, metrics_df, raw_data


# Load system assets
try:
    imputer, scaler, iqr_bounds, models, metrics_df, raw_data = get_cached_artifacts()
except Exception as e:
    st.error(f"Error loading models or dataset: {e}")
    st.info("Please make sure you have run 'py -3.10 src/model_training.py' to generate artifacts.")
    st.stop()


# Sidebar Configuration
st.sidebar.image("https://img.icons8.com/color/96/medical-heart.png", width=70)
st.sidebar.title("Configuration")

model_options = list(models.keys())
default_index = model_options.index("Weighted Soft Voting Ensemble (Proposed)") if "Weighted Soft Voting Ensemble (Proposed)" in model_options else 0
selected_model_name = st.sidebar.selectbox("Select ML Classifier", model_options, index=default_index)

# Display selected model metrics in sidebar
selected_row = metrics_df[metrics_df['Model'] == selected_model_name]
if not selected_row.empty:
    st.sidebar.markdown(f"**Model Performance Metrics:**")
    st.sidebar.markdown(f"- **Test Accuracy:** `{selected_row['Test Accuracy'].values[0] * 100:.1f}%`")
    st.sidebar.markdown(f"- **Test ROC-AUC:** `{selected_row['Test ROC-AUC'].values[0]:.3f}`")
    st.sidebar.markdown(f"- **Test F1-Score:** `{selected_row['Test F1-Score'].values[0]:.3f}`")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚡ Quick Patient Profiles")
col_p1, col_p2 = st.sidebar.columns(2)

if 'preset_loaded' not in st.session_state:
    st.session_state['preset_loaded'] = None

if col_p1.button("Healthy Profile", use_container_width=True):
    st.session_state['preset_loaded'] = "healthy"
if col_p2.button("High-Risk Profile", use_container_width=True):
    st.session_state['preset_loaded'] = "diabetic"

st.sidebar.markdown("---")
st.sidebar.caption("AML Project: Diabetes Prediction Using Machine Learning & Ensembling")


# Main Dashboard Header
st.markdown('<div class="main-header">🩺 Diabetes Risk Prediction & Clinical Decision Support</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated screening and risk stratification powered by Machine Learning and Weighted Soft Voting Ensembles.</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🩺 Patient Risk Assessment",
    "📁 Multi-Patient Screening",
    "📊 Model Benchmarks & Metrics",
    "🔬 Exploratory Data Analysis (EDA)",
    "📑 Methodology & Documentation"
])

# ==========================================
# TAB 1: INDIVIDUAL PATIENT RISK ASSESSMENT
# ==========================================
with tab1:
    st.markdown("### Enter Patient Clinical & Demographic Parameters")
    st.caption("Values are normalized and preprocessed according to clinical bounds and the IEEE reference methodology.")

    # Apply Presets if selected
    preset = st.session_state.get('preset_loaded', None)
    if preset == "healthy":
        p_defaults = {'p': 1, 'g': 88, 'bp': 68, 'st': 22, 'ins': 85, 'bmi': 22.8, 'dpf': 0.25, 'age': 25}
    elif preset == "diabetic":
        p_defaults = {'p': 5, 'g': 168, 'bp': 86, 'st': 34, 'ins': 220, 'bmi': 37.4, 'dpf': 0.82, 'age': 54}
    else:
        p_defaults = {'p': 2, 'g': 120, 'bp': 74, 'st': 25, 'ins': 105, 'bmi': 28.5, 'dpf': 0.42, 'age': 38}

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input(
            "Number of Pregnancies",
            min_value=0, max_value=20, value=p_defaults['p'], step=1,
            help="Number of times pregnant"
        )
        glucose = st.slider(
            "Plasma Glucose Concentration (mg/dL)",
            min_value=40.0, max_value=260.0, value=float(p_defaults['g']), step=1.0,
            help="Oral glucose tolerance test plasma glucose (Normal fasting: <100 mg/dL, Pre-diabetes: 100-125, Diabetic: ≥126)"
        )
        blood_pressure = st.slider(
            "Diastolic Blood Pressure (mm Hg)",
            min_value=40.0, max_value=140.0, value=float(p_defaults['bp']), step=1.0,
            help="Normal diastolic pressure is typically < 80 mm Hg"
        )
        skin_thickness = st.slider(
            "Triceps Skinfold Thickness (mm)",
            min_value=5.0, max_value=80.0, value=float(p_defaults['st']), step=1.0,
            help="Measures subcutaneous body fat"
        )

    with col2:
        insulin = st.slider(
            "2-Hour Serum Insulin (μU/mL)",
            min_value=10.0, max_value=700.0, value=float(p_defaults['ins']), step=1.0,
            help="Normal fasting serum insulin is between 15-150 μU/mL"
        )
        bmi = st.slider(
            "Body Mass Index - BMI (kg/m²)",
            min_value=12.0, max_value=60.0, value=float(p_defaults['bmi']), step=0.1,
            help="Weight in kg / (height in m)². Normal: 18.5 - 24.9, Overweight: 25 - 29.9, Obese: ≥30"
        )
        dpf = st.slider(
            "Diabetes Pedigree Function (DPF)",
            min_value=0.05, max_value=2.50, value=float(p_defaults['dpf']), step=0.01,
            help="Genetic risk score based on family history of diabetes"
        )
        age = st.slider(
            "Patient Age (years)",
            min_value=18, max_value=100, value=int(p_defaults['age']), step=1,
            help="Age of patient in years"
        )

    patient_dict = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }

    st.markdown("<br>", unsafe_allow_html=True)
    assess_button = st.button("🔍 Assess Diabetes Risk", type="primary", use_container_width=True)

    if assess_button or preset is not None:
        model = models[selected_model_name]
        X_proc = preprocess_patient(patient_dict, imputer, scaler, iqr_bounds)
        pred_label = int(model.predict(X_proc)[0])
        pred_prob = float(model.predict_proba(X_proc)[0, 1])

        # Risk Classification
        if pred_prob < 0.35:
            risk_tier = "Low Risk"
            badge_class = "badge-low"
            color_hex = "#16A34A"
        elif pred_prob < 0.65:
            risk_tier = "Moderate Risk"
            badge_class = "badge-moderate"
            color_hex = "#D97706"
        else:
            risk_tier = "High Risk"
            badge_class = "badge-high"
            color_hex = "#DC2626"

        st.markdown("### Diagnostic Assessment Report")

        if pred_label == 1:
            st.markdown(f"""
            <div class="card-diabetic">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h2 style="color: #DC2626; margin: 0;">⚠️ Diabetes Positive Risk Indicated</h2>
                    <span class="metric-badge {badge_class}">{risk_tier}</span>
                </div>
                <p style="margin-top: 10px; font-size: 1.1rem; color: #1F2937;">
                    The machine learning model (<strong>{selected_model_name}</strong>) predicts that the patient is 
                    <strong>likely to test positive for diabetes</strong> with an estimated risk probability of 
                    <strong style="color: #DC2626;">{pred_prob * 100:.1f}%</strong>.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="card-nondiabetic">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h2 style="color: #16A34A; margin: 0;">✅ Non-Diabetic Profile Indicated</h2>
                    <span class="metric-badge {badge_class}">{risk_tier}</span>
                </div>
                <p style="margin-top: 10px; font-size: 1.1rem; color: #1F2937;">
                    The machine learning model (<strong>{selected_model_name}</strong>) predicts that the patient is 
                    <strong>unlikely to have diabetes</strong>. Estimated disease probability is 
                    <strong style="color: #16A34A;">{pred_prob * 100:.1f}%</strong>.
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Visual Probability Bar
        st.markdown(f"**Predicted Probability of Diabetes:** `{pred_prob * 100:.1f}%`")
        st.progress(pred_prob)

        # Clinical Risk Factors Analysis
        st.markdown("#### Clinical Indicators Breakdown")
        col_c1, col_c2, col_c3 = st.columns(3)

        # Glucose status
        with col_c1:
            if glucose >= 126:
                st.error(f"🩸 **Glucose: {glucose} mg/dL**\nDiabetic Fasting Range (≥126)")
            elif glucose >= 100:
                st.warning(f"🩸 **Glucose: {glucose} mg/dL**\nPre-diabetic Range (100-125)")
            else:
                st.success(f"🩸 **Glucose: {glucose} mg/dL**\nNormal Range (<100)")

        # BMI status
        with col_c2:
            if bmi >= 30:
                st.error(f"⚖️ **BMI: {bmi:.1f} kg/m²**\nObese Range (≥30)")
            elif bmi >= 25:
                st.warning(f"⚖️ **BMI: {bmi:.1f} kg/m²**\nOverweight Range (25-29.9)")
            else:
                st.success(f"⚖️ **BMI: {bmi:.1f} kg/m²**\nHealthy Weight (18.5-24.9)")

        # Blood Pressure status
        with col_c3:
            if blood_pressure >= 90:
                st.error(f"🫀 **BP: {blood_pressure} mm Hg**\nStage 2 Hypertension (≥90)")
            elif blood_pressure >= 80:
                st.warning(f"🫀 **BP: {blood_pressure} mm Hg**\nStage 1 Hypertension (80-89)")
            else:
                st.success(f"🫀 **BP: {blood_pressure} mm Hg**\nNormal Diastolic (<80)")

        # Medical guidance note
        st.info("💡 **Clinical Disclaimer**: This AI prediction tool is intended for preliminary screening and educational decision support. Confirmatory clinical diagnosis requires laboratory tests (e.g., HbA1c, Oral Glucose Tolerance Test) conducted by licensed medical personnel.")


# ==========================================
# TAB 2: MULTI-PATIENT SCREENING
# ==========================================
with tab2:
    st.markdown("### Multi-Patient Clinical Screening from CSV")
    st.write("Upload a CSV file containing patient parameters to screen multiple patients simultaneously.")

    # Provide download for sample CSV template
    sample_df = pd.DataFrame([
        {'Pregnancies': 1, 'Glucose': 89, 'BloodPressure': 66, 'SkinThickness': 23, 'Insulin': 94, 'BMI': 28.1, 'DiabetesPedigreeFunction': 0.167, 'Age': 21},
        {'Pregnancies': 5, 'Glucose': 166, 'BloodPressure': 72, 'SkinThickness': 19, 'Insulin': 175, 'BMI': 35.8, 'DiabetesPedigreeFunction': 0.587, 'Age': 51},
        {'Pregnancies': 2, 'Glucose': 110, 'BloodPressure': 70, 'SkinThickness': 20, 'Insulin': 100, 'BMI': 24.5, 'DiabetesPedigreeFunction': 0.312, 'Age': 30}
    ])
    csv_sample_data = sample_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Sample CSV Template", csv_sample_data, "diabetes_patient_sample.csv", "text/csv")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            missing_cols = [c for c in FEATURE_NAMES if c not in batch_df.columns]
            if missing_cols:
                st.error(f"Uploaded CSV is missing required columns: {missing_cols}")
            else:
                st.success(f"Successfully loaded {len(batch_df)} patient records!")

                # Process all rows
                model = models[selected_model_name]
                results = []
                for _, row in batch_df.iterrows():
                    p_row = row[FEATURE_NAMES].to_dict()
                    X_scaled = preprocess_patient(p_row, imputer, scaler, iqr_bounds)
                    p_pred = int(model.predict(X_scaled)[0])
                    p_prob = float(model.predict_proba(X_scaled)[0, 1])

                    tier = "High" if p_prob >= 0.65 else ("Moderate" if p_prob >= 0.35 else "Low")
                    results.append({
                        'Prediction': 'Diabetic' if p_pred == 1 else 'Non-Diabetic',
                        'Diabetes_Probability': round(p_prob, 4),
                        'Risk_Tier': tier
                    })

                res_df = pd.concat([batch_df, pd.DataFrame(results)], axis=1)

                # Summary metrics
                c_t1, c_t2, c_t3 = st.columns(3)
                total_patients = len(res_df)
                diabetic_count = sum(res_df['Prediction'] == 'Diabetic')
                high_risk_count = sum(res_df['Risk_Tier'] == 'High')

                c_t1.metric("Total Patients Screened", total_patients)
                c_t2.metric("Predicted Diabetic Cases", f"{diabetic_count} ({diabetic_count/total_patients*100:.1f}%)")
                c_t3.metric("High-Risk Stratified", f"{high_risk_count} ({high_risk_count/total_patients*100:.1f}%)")

                st.dataframe(res_df.head(20), use_container_width=True)

                # Export Results
                csv_export = res_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Complete Patient Screening Predictions (CSV)",
                    csv_export,
                    "patient_diabetes_predictions.csv",
                    "text/csv",
                    type="primary"
                )
        except Exception as e:
            st.error(f"Error processing CSV: {e}")


# ==========================================
# TAB 3: MODEL BENCHMARKS & METRICS
# ==========================================
with tab3:
    st.markdown("### Comprehensive Machine Learning Benchmark")
    st.write("Comparing all 8 individual classifiers, Stacking, and the IEEE Access Weighted Soft Voting Ensemble.")

    # Format and display table
    st.dataframe(metrics_df, use_container_width=True)

    st.markdown("---")
    st.markdown("### Visual Performance Comparison")

    col_fig1, col_fig2 = st.columns(2)
    with col_fig1:
        if os.path.exists("reports/figures/roc_curves.png"):
            st.image("reports/figures/roc_curves.png", caption="ROC Curves & AUC Comparison (Higher is Better)")

    with col_fig2:
        if os.path.exists("reports/figures/model_comparison.png"):
            st.image("reports/figures/model_comparison.png", caption="Accuracy, F1-Score & ROC-AUC Comparison")

    st.markdown("---")
    col_fig3, col_fig4 = st.columns(2)
    with col_fig3:
        if os.path.exists("reports/figures/feature_importance.png"):
            st.image("reports/figures/feature_importance.png", caption="Clinical Feature Importance Drivers")

    with col_fig4:
        if os.path.exists("reports/figures/confusion_matrices.png"):
            st.image("reports/figures/confusion_matrices.png", caption="Confusion Matrices Across All Classifiers")


# ==========================================
# TAB 4: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
with tab4:
    st.markdown("### Exploratory Data Analysis of Pima Indians Diabetes Dataset")
    
    st.markdown("#### Dataset Statistical Overview")
    st.dataframe(raw_data.describe().round(2), use_container_width=True)

    col_eda1, col_eda2 = st.columns(2)
    with col_eda1:
        st.markdown("#### Outcome Class Distribution")
        fig_pie, ax_pie = plt.subplots(figsize=(5, 5))
        counts = raw_data['Outcome'].value_counts()
        ax_pie.pie(
            counts, labels=['Non-Diabetic (0)', 'Diabetic (1)'],
            autopct='%1.1f%%', colors=['#2b5c8f', '#e76f51'],
            startangle=140, explode=[0, 0.06]
        )
        ax_pie.set_title("Target Class Balance (N=768)")
        st.pyplot(fig_pie)
        plt.close(fig_pie)

    with col_eda2:
        if os.path.exists("reports/figures/correlation_heatmap.png"):
            st.image("reports/figures/correlation_heatmap.png", caption="Feature Correlation Heatmap with Outcome")

    st.markdown("---")
    st.markdown("#### Interactive Feature Distribution Explorer")
    selected_feature = st.selectbox("Select Feature to Visualize Distribution:", FEATURE_NAMES, index=1)

    fig_hist, ax_hist = plt.subplots(figsize=(10, 4.5))
    sns.kdeplot(data=raw_data[raw_data['Outcome'] == 0][selected_feature], label="Non-Diabetic (0)", fill=True, color="#2b5c8f", alpha=0.5, ax=ax_hist)
    sns.kdeplot(data=raw_data[raw_data['Outcome'] == 1][selected_feature], label="Diabetic (1)", fill=True, color="#e76f51", alpha=0.5, ax=ax_hist)
    ax_hist.set_title(f"Kernel Density Estimate for '{selected_feature}' by Diabetes Status", fontsize=12, fontweight='bold')
    ax_hist.set_xlabel(selected_feature)
    ax_hist.set_ylabel("Density")
    ax_hist.legend()
    st.pyplot(fig_hist)
    plt.close(fig_hist)


# ==========================================
# TAB 5: METHODOLOGY & DOCUMENTATION
# ==========================================
with tab5:
    st.markdown("### 🔬 Applied Machine Learning Architecture & Methodology")
    st.markdown("""
    #### 1. Core Classification Algorithms Implemented
    The system evaluates all 5 primary machine learning models plus an advanced ensembling framework:
    - **Logistic Regression**: Linear boundary baseline for estimating log-odds of diabetic probability.
    - **Decision Tree Classifier**: Interpretable tree structure partitioning feature space using Gini impurity.
    - **Random Forest Classifier**: Bagging ensemble of de-correlated decision trees mitigating variance and overfitting.
    - **Support Vector Machine (SVM)**: Radial Basis Function (RBF) kernel mapping inputs into high-dimensional space for non-linear separation.
    - **K-Nearest Neighbors (KNN)**: Non-parametric distance-based classification evaluating neighbor proximity.
    - **Weighted Soft Voting Ensemble**: Formulation from IEEE Access (2020) weighting probabilistic outputs by each model's cross-validated ROC-AUC ($W_j$).

    ---

    #### 2. Data Cleaning & Zero Treatment
    In clinical datasets like Pima Indians, physiological measurements such as **Glucose, Blood Pressure, Skin Thickness, Insulin, and BMI** cannot biologically be zero for a living patient:
    - Zeroes in these continuous attributes represent **missing values**.
    - These zeros were converted to `NaN` and imputed using **median imputation** (Equation 3 in paper).

    #### 3. Outlier Rejection via Interquartile Range (IQR)
    To eliminate distortion from measurement anomalies, continuous attributes were filtered using the **IQR method** (Equation 2 in paper):
    $$\\text{IQR} = Q_3 - Q_1$$
    $$\\text{Valid Range} = [Q_1 - 1.5 \\times \\text{IQR}, \\; Q_3 + 1.5 \\times \\text{IQR}]$$

    #### 4. Feature Standardization
    All 8 clinical features were scaled using $Z$-score normalization:
    $$Z = \\frac{X - \\mu}{\\sigma}$$
    where $\\mu$ is the feature mean and $\\sigma$ is the standard deviation.

    #### 5. Weighted Soft Voting Ensemble Formulation
    Individual models yield predicted class probabilities $P_{ij}$ for each class $i \\in \\{0, 1\\}$ and classifier $j$.
    The ensembled confidence $P^{en}_i$ is computed using the **ROC-AUC weighted aggregation** (Equation 6 in paper):
    $$P^{en}_i = \\frac{\\sum_{j=1}^{m} (W_j \\times P_{ij})}{\\sum_{i=1}^{C} \\sum_{j=1}^{m} (W_j \\times P_{ij})}$$
    where $W_j$ represents the 5-Fold Cross-Validation ROC-AUC of the $j$-th classifier.
    """)

