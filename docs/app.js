// DiaPredict - Clinical Machine Learning Inference & Decision Support Engine
(function() {
  const models = window.DIABETES_ML_MODELS;
  if (!models) {
    console.error("Clinical model data not found.");
    return;
  }

  // --- State & Storage Keys ---
  const STORAGE_KEY_HISTORY = "diapredict_history_records";
  const STORAGE_KEY_USER = "diapredict_active_user";

  let latestAssessment = null;

  // --- Navigation & Tab Switching ---
  const navLinks = document.querySelectorAll(".nav-tab");
  const tabPanes = document.querySelectorAll(".tab-pane");

  window.switchTab = function(tabId) {
    navLinks.forEach(link => {
      if (link.dataset.tab === tabId) {
        link.classList.add("active");
      } else {
        link.classList.remove("active");
      }
    });

    tabPanes.forEach(pane => {
      if (pane.id === tabId) {
        pane.classList.add("active");
        window.scrollTo({ top: 0, behavior: 'smooth' });
      } else {
        pane.classList.remove("active");
      }
    });

    if (tabId === "dashboard-tab") {
      updateDashboardStats();
    } else if (tabId === "history-tab") {
      renderHistoryTable();
    } else if (tabId === "profile-tab") {
      updateProfileView();
    }
  };

  navLinks.forEach(link => {
    link.addEventListener("click", function(e) {
      e.preventDefault();
      switchTab(this.dataset.tab);
    });
  });

  // --- Quick Sample Fill ---
  const samples = {
    normal: {
      name: "Sarah Jenkins (Healthy Baseline)",
      pregnancies: 1,
      glucose: 85,
      bloodPressure: 68,
      skinThickness: 24,
      insulin: 80,
      bmi: 22.4,
      pedigree: 0.235,
      age: 27
    },
    risk: {
      name: "Eleanor Vance (High Risk Profile)",
      pregnancies: 6,
      glucose: 154,
      bloodPressure: 78,
      skinThickness: 34,
      insulin: 140,
      bmi: 34.2,
      pedigree: 0.655,
      age: 52
    }
  };

  window.fillSample = function(type) {
    const s = samples[type];
    if (!s) return;
    document.getElementById("patient_name").value = s.name;
    document.getElementById("pregnancies").value = s.pregnancies;
    document.getElementById("glucose").value = s.glucose;
    document.getElementById("blood_pressure").value = s.bloodPressure;
    document.getElementById("skin_thickness").value = s.skinThickness;
    document.getElementById("insulin").value = s.insulin;
    document.getElementById("bmi").value = s.bmi;
    document.getElementById("diabetes_pedigree").value = s.pedigree;
    document.getElementById("age").value = s.age;
    switchTab("predict-tab");
  };

  // --- ML Inference Pipeline ---
  function preprocess(inputVals) {
    // 8 features: [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, Pedigree, Age]
    // Zero imputation on biological zeros: indices 1, 2, 3, 4, 5
    const zeroIndices = [1, 2, 3, 4, 5];
    const imputed = inputVals.map((v, i) => {
      if (zeroIndices.includes(i) && (v === 0 || isNaN(v))) {
        return models.imputer_stats[i];
      }
      return isNaN(v) ? models.imputer_stats[i] : v;
    });

    // StandardScaler: (x - mean) / scale
    const scaled = imputed.map((v, i) => (v - models.scaler_mean[i]) / models.scaler_scale[i]);
    return { imputed, scaled };
  }

  function evalTree(node, scaledFeats) {
    if (node.leaf) {
      return { pred: node.pred, prob: node.prob };
    }
    if (scaledFeats[node.f] <= node.th) {
      return evalTree(node.l, scaledFeats);
    } else {
      return evalTree(node.r, scaledFeats);
    }
  }

  function predictDecisionTree(scaledFeats) {
    return evalTree(models.decision_tree, scaledFeats);
  }

  function predictLogisticRegression(scaledFeats) {
    const lr = models.logistic_regression;
    let z = lr.intercept;
    for (let i = 0; i < 8; i++) {
      z += scaledFeats[i] * lr.coef[i];
    }
    const prob = 1.0 / (1.0 + Math.exp(-z));
    return { pred: prob >= 0.5 ? 1 : 0, prob: Math.round(prob * 10000) / 10000 };
  }

  function predictRandomForest(scaledFeats) {
    const trees = models.random_forest_trees;
    let sumProb = 0;
    for (let i = 0; i < trees.length; i++) {
      sumProb += evalTree(trees[i], scaledFeats).prob;
    }
    const avgProb = sumProb / trees.length;
    return { pred: avgProb >= 0.5 ? 1 : 0, prob: Math.round(avgProb * 10000) / 10000 };
  }

  // --- Prediction Form Submission with Animated Loading ---
  const predictForm = document.getElementById("predict-form");
  const submitBtn = document.getElementById("predict-submit-btn");
  const resultSection = document.getElementById("result-section");

  if (predictForm) {
    predictForm.addEventListener("submit", function(e) {
      e.preventDefault();

      const nameInput = document.getElementById("patient_name").value.trim();
      const patientName = nameInput || "Patient #" + Math.floor(1000 + Math.random() * 9000);
      const patientId = "PID-2026-" + Math.floor(1000 + Math.random() * 9000);
      const modelChoice = document.getElementById("model_choice").value;

      const rawVals = [
        parseFloat(document.getElementById("pregnancies").value) || 0,
        parseFloat(document.getElementById("glucose").value) || 0,
        parseFloat(document.getElementById("blood_pressure").value) || 0,
        parseFloat(document.getElementById("skin_thickness").value) || 0,
        parseFloat(document.getElementById("insulin").value) || 0,
        parseFloat(document.getElementById("bmi").value) || 0,
        parseFloat(document.getElementById("diabetes_pedigree").value) || 0,
        parseFloat(document.getElementById("age").value) || 0
      ];

      // 1. Show Animated Loading State: "🔄 Analyzing Health Data..."
      submitBtn.classList.add("btn-loading");
      submitBtn.innerHTML = `<span class="btn-spinner"></span> 🔄 Analyzing Health Data...`;

      // 2. Realistic ML computation delay (~650ms)
      setTimeout(() => {
        const { imputed, scaled } = preprocess(rawVals);

        let res, modelLabel;
        if (modelChoice === "logistic_regression") {
          res = predictLogisticRegression(scaled);
          modelLabel = "Logistic Regression";
        } else if (modelChoice === "random_forest") {
          res = predictRandomForest(scaled);
          modelLabel = "Random Forest";
        } else {
          res = predictDecisionTree(scaled);
          modelLabel = "Decision Tree (Best Model)";
        }

        const now = new Date();
        const timestamp = now.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" }) +
                          " • " + now.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" });

        const assessmentData = {
          patientName,
          patientId,
          timestamp,
          rawVals,
          res,
          modelLabel
        };

        latestAssessment = assessmentData;

        // Render Comprehensive Result Dashboard
        renderResultDashboard(assessmentData);

        // Save into local history & update stats
        saveToHistory(assessmentData);
        updateDashboardStats();
        updateProfileView();

        // Restore button state
        submitBtn.classList.remove("btn-loading");
        submitBtn.innerHTML = `⚡ Run Real-Time ML Prediction`;

        // Smooth reveal & scroll to result
        resultSection.style.display = "block";
        resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 650);
    });
  }

  // --- Render Result Dashboard & Circular Gauge ---
  function renderResultDashboard(data) {
    const { patientName, patientId, timestamp, rawVals, res, modelLabel } = data;
    const probPct = Math.round(res.prob * 100);

    // Header Meta
    document.getElementById("res-patient-name").textContent = patientName;
    document.getElementById("res-patient-id").textContent = patientId;
    document.getElementById("res-timestamp").textContent = timestamp;
    document.getElementById("res-model-badge").textContent = modelLabel;

    // Print Header
    document.getElementById("print-patient-id").textContent = patientId;
    document.getElementById("print-timestamp").textContent = timestamp;

    // Circular Risk Gauge Animation
    const gaugeCircle = document.getElementById("gauge-circle");
    const gaugePercentText = document.getElementById("gauge-percent-text");
    const riskBadge = document.getElementById("res-risk-badge");
    const headline = document.getElementById("res-status-headline");
    const desc = document.getElementById("res-desc");

    // Circumference = 2 * PI * 70 = 439.82 (~440)
    const circumference = 440;
    const offset = circumference - (circumference * (probPct / 100));
    gaugeCircle.style.strokeDashoffset = offset;

    // Animate percentage counter in gauge
    animateCounter(gaugePercentText, probPct);

    // Set Risk Level (Low, Medium, High)
    headline.textContent = `Diabetes Risk: ${probPct}%`;

    if (probPct >= 65 || res.pred === 1) {
      gaugeCircle.style.stroke = "#ef4444";
      riskBadge.className = "risk-badge high";
      riskBadge.innerHTML = `● High Risk`;
      desc.textContent = `Evaluated clinical biomarkers show an elevated statistical likelihood of diabetes mellitus (${probPct}% probability). Immediate clinical laboratory screening recommended.`;
    } else if (probPct >= 35) {
      gaugeCircle.style.stroke = "#f59e0b";
      riskBadge.className = "risk-badge medium";
      riskBadge.innerHTML = `● Moderate Risk`;
      desc.textContent = `Evaluated physiological indicators reflect borderline/moderate diabetes risk factors (${probPct}% probability). Preventative metabolic adjustments advised.`;
    } else {
      gaugeCircle.style.stroke = "#10b981";
      riskBadge.className = "risk-badge low";
      riskBadge.innerHTML = `● Low Risk`;
      desc.textContent = `Evaluated physiological indicators reflect healthy physiological parameters (${probPct}% probability). Routine annual wellness monitoring recommended.`;
    }

    // --- 4 Clinical Biomarker Indicators ---
    const glucose = rawVals[1];
    const bp = rawVals[2];
    const bmi = rawVals[5];
    const age = rawVals[7];

    // 1. Glucose Indicator
    const gluElem = document.getElementById("ind-glucose-val");
    const gluStatus = document.getElementById("ind-glucose-status");
    gluElem.innerHTML = `${glucose} <span class="ind-unit">mg/dL</span>`;
    if (glucose >= 126) {
      gluStatus.className = "ind-status danger";
      gluStatus.textContent = "Diabetic Threshold";
    } else if (glucose >= 100) {
      gluStatus.className = "ind-status warning";
      gluStatus.textContent = "Pre-diabetic / Elevated";
    } else {
      gluStatus.className = "ind-status normal";
      gluStatus.textContent = "Normal Fasting";
    }

    // 2. BMI Indicator
    const bmiElem = document.getElementById("ind-bmi-val");
    const bmiStatus = document.getElementById("ind-bmi-status");
    bmiElem.innerHTML = `${bmi.toFixed(1)} <span class="ind-unit">kg/m²</span>`;
    if (bmi >= 30.0) {
      bmiStatus.className = "ind-status danger";
      bmiStatus.textContent = "Obesity Class";
    } else if (bmi >= 25.0) {
      bmiStatus.className = "ind-status warning";
      bmiStatus.textContent = "Overweight Tier";
    } else if (bmi >= 18.5) {
      bmiStatus.className = "ind-status normal";
      bmiStatus.textContent = "Healthy Weight";
    } else {
      bmiStatus.className = "ind-status warning";
      bmiStatus.textContent = "Underweight";
    }

    // 3. Blood Pressure Indicator
    const bpElem = document.getElementById("ind-bp-val");
    const bpStatus = document.getElementById("ind-bp-status");
    bpElem.innerHTML = `${bp} <span class="ind-unit">mm Hg</span>`;
    if (bp >= 90) {
      bpStatus.className = "ind-status danger";
      bpStatus.textContent = "Hypertension";
    } else if (bp >= 80) {
      bpStatus.className = "ind-status warning";
      bpStatus.textContent = "Pre-hypertension";
    } else {
      bpStatus.className = "ind-status normal";
      bpStatus.textContent = "Normal Diastolic";
    }

    // 4. Age Indicator
    const ageElem = document.getElementById("ind-age-val");
    const ageStatus = document.getElementById("ind-age-status");
    ageElem.innerHTML = `${age} <span class="ind-unit">Yrs</span>`;
    if (age >= 50) {
      ageStatus.className = "ind-status danger";
      ageStatus.textContent = "Elevated Age Factor";
    } else if (age >= 35) {
      ageStatus.className = "ind-status warning";
      ageStatus.textContent = "Moderate Age Risk";
    } else {
      ageStatus.className = "ind-status normal";
      ageStatus.textContent = "Standard Age Tier";
    }

    // --- Feature Contribution Breakdown ---
    // Relative importance weights adjusted based on patient deviations
    let wGlu = Math.max(15, Math.min(50, Math.round((glucose / 180) * 45)));
    let wBmi = Math.max(12, Math.min(35, Math.round((bmi / 40) * 28)));
    let wAge = Math.max(10, Math.min(25, Math.round((age / 70) * 18)));
    let wPed = Math.max(8, Math.min(20, Math.round((rawVals[6] / 1.2) * 14)));
    let wOther = Math.max(5, 100 - (wGlu + wBmi + wAge + wPed));

    const totalWeight = wGlu + wBmi + wAge + wPed + wOther;
    wGlu = Math.round((wGlu / totalWeight) * 100);
    wBmi = Math.round((wBmi / totalWeight) * 100);
    wAge = Math.round((wAge / totalWeight) * 100);
    wPed = Math.round((wPed / totalWeight) * 100);
    wOther = 100 - (wGlu + wBmi + wAge + wPed);

    document.getElementById("contrib-glucose").style.width = `${wGlu}%`;
    document.getElementById("contrib-glucose-txt").textContent = `${wGlu}%`;

    document.getElementById("contrib-bmi").style.width = `${wBmi}%`;
    document.getElementById("contrib-bmi-txt").textContent = `${wBmi}%`;

    document.getElementById("contrib-age").style.width = `${wAge}%`;
    document.getElementById("contrib-age-txt").textContent = `${wAge}%`;

    document.getElementById("contrib-pedigree").style.width = `${wPed}%`;
    document.getElementById("contrib-pedigree-txt").textContent = `${wPed}%`;

    document.getElementById("contrib-other").style.width = `${wOther}%`;
    document.getElementById("contrib-other-txt").textContent = `${wOther}%`;

    // --- Clinical Recommendations ---
    const recList = document.getElementById("res-recs");
    if (probPct >= 50) {
      recList.innerHTML = `
        <li><strong>Confirmatory Diagnostic Tests:</strong> Schedule an Oral Glucose Tolerance Test (OGTT) and Fasting Plasma Glucose (FPG) with a licensed physician.</li>
        <li><strong>HbA1c Evaluation:</strong> Order a laboratory Glycated Hemoglobin (HbA1c) test to determine glycemic control over the past 90 days.</li>
        <li><strong>Medical Nutrition Therapy:</strong> Transition towards an evidence-based Mediterranean or low-glycemic dietary regimen with reduced refined sugars.</li>
        <li><strong>Physical Exercise:</strong> Implement at least 150 minutes per week of moderate-intensity aerobic physical exercise (e.g. brisk walking).</li>
      `;
    } else {
      recList.innerHTML = `
        <li><strong>Preventative Health Maintenance:</strong> Sustain a balanced, nutrient-rich dietary profile abundant in dietary fiber and lean proteins.</li>
        <li><strong>Weight Management:</strong> Maintain a healthy BMI in the 18.5–24.9 kg/m² bracket through daily active movement.</li>
        <li><strong>Periodic Wellness Check:</strong> Undergo routine annual clinical wellness checkups and lipid / glucose screening.</li>
      `;
    }
  }

  function animateCounter(elem, target) {
    let current = 0;
    const step = Math.ceil(target / 25) || 1;
    const timer = setInterval(() => {
      current += step;
      if (current >= target) {
        current = target;
        clearInterval(timer);
      }
      elem.textContent = `${current}%`;
    }, 25);
  }

  window.startNewPrediction = function() {
    resultSection.style.display = "none";
    predictForm.reset();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  // --- Download Report Action ---
  window.downloadReport = function() {
    if (!latestAssessment) {
      alert("Please run a patient prediction first before downloading the report.");
      return;
    }
    // Triggers printable clinical report view formatted via @media print in style.css
    window.print();
  };

  // --- History Management ---
  function getHistory() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY_HISTORY);
      return raw ? JSON.parse(raw) : [];
    } catch(e) {
      return [];
    }
  }

  function saveToHistory(assessment) {
    const list = getHistory();
    list.unshift({
      id: assessment.patientId,
      name: assessment.patientName,
      timestamp: assessment.timestamp,
      model: assessment.modelLabel,
      glucose: assessment.rawVals[1],
      bmi: assessment.rawVals[5],
      age: assessment.rawVals[7],
      prob: Math.round(assessment.res.prob * 100),
      pred: assessment.res.pred
    });

    if (list.length > 50) list.pop();
    try {
      localStorage.setItem(STORAGE_KEY_HISTORY, JSON.stringify(list));
    } catch(e) {}
    renderHistoryTable();
  }

  function renderHistoryTable() {
    const list = getHistory();
    const countElem = document.getElementById("history-count");
    if (countElem) countElem.textContent = list.length;

    const tbody = document.getElementById("history-tbody");
    if (!tbody) return;

    if (list.length === 0) {
      tbody.innerHTML = `<tr><td colspan="8" style="text-align:center; padding:30px; color:#8fa5bd;">No assessments logged yet. Run a prediction to record entries.</td></tr>`;
      return;
    }

    tbody.innerHTML = list.map(item => {
      const isHigh = item.prob >= 65 || item.pred === 1;
      const badgeClass = isHigh ? "badge danger" : (item.prob >= 35 ? "badge warning" : "badge safe");
      const badgeLabel = isHigh ? "High Risk" : (item.prob >= 35 ? "Moderate" : "Low Risk");
      return `
        <tr>
          <td><strong>${escapeHtml(item.name)}</strong><br><small style="color:#64748b;">${item.id}</small></td>
          <td>${item.timestamp}</td>
          <td><span style="color:#38bdf8;">${item.model}</span></td>
          <td>${item.glucose} mg/dL</td>
          <td>${Number(item.bmi).toFixed(1)}</td>
          <td>${item.age}</td>
          <td><strong>${item.prob}%</strong></td>
          <td><span class="${badgeClass}">${badgeLabel}</span></td>
        </tr>
      `;
    }).join("");
  }

  window.clearHistory = function() {
    if (confirm("Are you sure you want to clear all screening history records?")) {
      localStorage.removeItem(STORAGE_KEY_HISTORY);
      renderHistoryTable();
      updateDashboardStats();
      updateProfileView();
    }
  };

  // --- Dashboard Tab Statistics ---
  function updateDashboardStats() {
    const list = getHistory();
    const total = list.length;
    let high = 0;
    let low = 0;
    let sumProb = 0;

    list.forEach(item => {
      sumProb += item.prob;
      if (item.prob >= 65 || item.pred === 1) high++;
      else low++;
    });

    const avg = total > 0 ? Math.round(sumProb / total) : 0;

    const totalElem = document.getElementById("dash-total-count");
    const highElem = document.getElementById("dash-high-count");
    const lowElem = document.getElementById("dash-low-count");
    const avgElem = document.getElementById("dash-avg-risk");

    if (totalElem) totalElem.textContent = total;
    if (highElem) highElem.textContent = high;
    if (lowElem) lowElem.textContent = low;
    if (avgElem) avgElem.textContent = `${avg}%`;

    const dashContainer = document.getElementById("dash-latest-container");
    if (dashContainer) {
      if (latestAssessment) {
        const prob = Math.round(latestAssessment.res.prob * 100);
        const isHigh = prob >= 65 || latestAssessment.res.pred === 1;
        dashContainer.innerHTML = `
          <div class="dashboard-card" style="margin-top: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1c3855; padding-bottom: 15px; margin-bottom: 18px; flex-wrap: wrap; gap: 10px;">
              <div>
                <span class="eyebrow">MOST RECENT CLINICAL ASSESSMENT</span>
                <h2 style="margin: 4px 0 0; font-size: 22px;">${escapeHtml(latestAssessment.patientName)} (${latestAssessment.patientId})</h2>
              </div>
              <span class="badge ${isHigh ? 'danger' : 'safe'}" style="font-size: 13px; padding: 6px 14px;">
                ${isHigh ? 'High Risk Assessment' : 'Low Risk Assessment'}
              </span>
            </div>
            <div style="display: flex; align-items: center; gap: 30px; flex-wrap: wrap;">
              <div style="font-size: 38px; font-weight: 800; color: ${isHigh ? '#f87171' : '#34d399'};">
                ${prob}% <span style="font-size: 14px; color: #8fa5bd; font-weight: 500;">Risk Score</span>
              </div>
              <div style="color: #9bb1c9; font-size: 13.5px; line-height: 1.6; flex: 1;">
                Evaluated using <strong>${latestAssessment.modelLabel}</strong> on ${latestAssessment.timestamp}. Glucose: ${latestAssessment.rawVals[1]} mg/dL • BMI: ${latestAssessment.rawVals[5]} kg/m² • Age: ${latestAssessment.rawVals[7]}.
              </div>
              <button type="button" class="btn primary" onclick="switchTab('predict-tab'); resultSection.style.display='block'; resultSection.scrollIntoView({behavior:'smooth'});">
                Inspect Result →
              </button>
            </div>
          </div>
        `;
      }
    }
  }

  // --- Auth Modal & Profile Management ---
  let authMode = 'login'; // 'login' or 'register'

  window.openAuthModal = function() {
    const modal = document.getElementById("auth-modal");
    if (modal) modal.classList.add("active");
  };

  window.closeAuthModal = function() {
    const modal = document.getElementById("auth-modal");
    if (modal) modal.classList.remove("active");
  };

  window.handleModalOverlayClick = function(e) {
    if (e.target.id === "auth-modal") {
      closeAuthModal();
    }
  };

  window.setAuthMode = function(mode) {
    authMode = mode;
    const btnLogin = document.getElementById("tab-btn-login");
    const btnReg = document.getElementById("tab-btn-register");
    const nameGroup = document.getElementById("reg-name-group");
    const title = document.getElementById("auth-title");
    const sub = document.getElementById("auth-subtitle");
    const submit = document.getElementById("auth-submit-btn");

    if (mode === 'register') {
      btnLogin.classList.remove("active");
      btnReg.classList.add("active");
      nameGroup.style.display = "block";
      title.textContent = "Create Clinician Account";
      sub.textContent = "Register a new practitioner or patient profile.";
      submit.textContent = "Create Account";
    } else {
      btnReg.classList.remove("active");
      btnLogin.classList.add("active");
      nameGroup.style.display = "none";
      title.textContent = "Sign In to DiaPredict";
      sub.textContent = "Access your saved clinical predictions and profile.";
      submit.textContent = "Sign In";
    }
  };

  window.handleAuthSubmit = function(e) {
    e.preventDefault();
    const email = document.getElementById("auth-email").value.trim();
    let name = "Clinician User";

    if (authMode === 'register') {
      name = document.getElementById("auth-fullname").value.trim() || email.split("@")[0];
    } else {
      name = email.split("@")[0];
      name = name.charAt(0).toUpperCase() + name.slice(1);
    }

    const userData = {
      name: name,
      email: email,
      role: "Clinical Practitioner • Screening Analyst",
      authenticated: true,
      lastLogin: new Date().toLocaleDateString()
    };

    localStorage.setItem(STORAGE_KEY_USER, JSON.stringify(userData));
    closeAuthModal();
    updateProfileView();
    alert(`Welcome, ${name}! Signed in successfully.`);
  };

  window.logoutUser = function() {
    localStorage.removeItem(STORAGE_KEY_USER);
    updateProfileView();
    alert("You have logged out.");
  };

  function updateProfileView() {
    let user;
    try {
      const raw = localStorage.getItem(STORAGE_KEY_USER);
      user = raw ? JSON.parse(raw) : null;
    } catch(e) {
      user = null;
    }

    const navAuthBtn = document.getElementById("nav-auth-btn");
    const profName = document.getElementById("prof-name");
    const profEmail = document.getElementById("prof-email");
    const profAvatar = document.getElementById("prof-avatar");
    const profAuth = document.getElementById("prof-auth-status");
    const profCount = document.getElementById("prof-eval-count");

    const historyList = getHistory();
    if (profCount) profCount.textContent = `${historyList.length} Assessments`;

    if (user && user.authenticated) {
      if (navAuthBtn) navAuthBtn.textContent = `👤 ${user.name}`;
      if (profName) profName.textContent = user.name;
      if (profEmail) profEmail.textContent = user.email;
      if (profAvatar) profAvatar.textContent = user.name.charAt(0).toUpperCase();
      if (profAuth) {
        profAuth.textContent = "Active Practitioner Session";
        profAuth.style.color = "#34d399";
      }
    } else {
      if (navAuthBtn) navAuthBtn.textContent = "🔐 Login/Register";
      if (profName) profName.textContent = "Guest Practitioner";
      if (profEmail) profEmail.textContent = "guest@diapredict.org";
      if (profAvatar) profAvatar.textContent = "👤";
      if (profAuth) {
        profAuth.textContent = "Guest Mode (Local Storage)";
        profAuth.style.color = "#8fa5bd";
      }
    }
  }

  function escapeHtml(str) {
    if (!str) return "";
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  // --- Initial Mount ---
  renderHistoryTable();
  updateDashboardStats();
  updateProfileView();

  console.log("DiaPredict Engine v2.0 Initialized successfully.");
})();
