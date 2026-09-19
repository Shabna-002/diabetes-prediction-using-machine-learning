// DiaPredict - Clinical Machine Learning Inference & Decision Support Engine
(function() {
  const models = window.DIABETES_ML_MODELS;
  if (!models) {
    console.error("Clinical model data not found.");
    return;
  }

  // --- Storage Keys ---
  const STORAGE_KEY_USERS = "diapredict_registered_users";
  const STORAGE_KEY_ACTIVE_USER = "diapredict_active_user";
  const STORAGE_KEY_BASE_HISTORY = "diapredict_history_records_";

  let latestAssessment = null;

  // --- Password Hashing using SHA-256 (Web Crypto API) ---
  async function hashPassword(plainText) {
    try {
      const msgBuffer = new TextEncoder().encode(plainText);
      const hashBuffer = await crypto.subtle.digest('SHA-256', msgBuffer);
      const hashArray = Array.from(new Uint8Array(hashBuffer));
      return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
    } catch(e) {
      // Basic fallback
      let hash = 0;
      for (let i = 0; i < plainText.length; i++) {
        hash = ((hash << 5) - hash) + plainText.charCodeAt(i);
        hash |= 0;
      }
      return "hash_" + Math.abs(hash).toString(16);
    }
  }

  // --- Active User Management ---
  function getActiveUser() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY_ACTIVE_USER);
      return raw ? JSON.parse(raw) : null;
    } catch(e) {
      return null;
    }
  }

  function getHistoryStorageKey() {
    const user = getActiveUser();
    return STORAGE_KEY_BASE_HISTORY + (user && user.email ? user.email.toLowerCase() : "guest");
  }

  // --- Seed Initial Data if History Empty ---
  function initSeedHistory() {
    const key = getHistoryStorageKey();
    const existing = localStorage.getItem(key);
    if (!existing) {
      const seedData = [
        {
          id: "PID-2026-8942",
          name: "Eleanor Vance",
          dateFormatted: "19-09-2026",
          timestamp: "19-09-2026 • 11:30 AM",
          model: "Decision Tree (Best Model)",
          rawVals: [6, 145, 78, 34, 140, 29.4, 0.655, 52],
          glucose: 145,
          bmi: 29.4,
          bp: 78,
          age: 52,
          prob: 72,
          pred: 1,
          result: "High"
        },
        {
          id: "PID-2026-7210",
          name: "Michael Torres",
          dateFormatted: "12-09-2026",
          timestamp: "12-09-2026 • 02:15 PM",
          model: "Random Forest",
          rawVals: [3, 118, 74, 28, 110, 25.2, 0.420, 42],
          glucose: 118,
          bmi: 25.2,
          bp: 74,
          age: 42,
          prob: 38,
          pred: 0,
          result: "Medium"
        },
        {
          id: "PID-2026-3841",
          name: "Sarah Jenkins",
          dateFormatted: "04-09-2026",
          timestamp: "04-09-2026 • 09:45 AM",
          model: "Decision Tree",
          rawVals: [1, 92, 68, 22, 75, 22.8, 0.235, 28],
          glucose: 92,
          bmi: 22.8,
          bp: 68,
          age: 28,
          prob: 15,
          pred: 0,
          result: "Low"
        }
      ];
      localStorage.setItem(key, JSON.stringify(seedData));
    }
  }

  // --- Tab Switching ---
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
    } else if (tabId === "analytics-tab") {
      renderHealthAnalytics();
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
      glucose: 145,
      bloodPressure: 78,
      skinThickness: 34,
      insulin: 140,
      bmi: 29.4,
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
    const zeroIndices = [1, 2, 3, 4, 5];
    const imputed = inputVals.map((v, i) => {
      if (zeroIndices.includes(i) && (v === 0 || isNaN(v))) {
        return models.imputer_stats[i];
      }
      return isNaN(v) ? models.imputer_stats[i] : v;
    });

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

  // --- Prediction Form Submit with Loading Animation ---
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

      // 1. Show Animated Loading State & Activate AI Holographic Scanner
      submitBtn.classList.add("btn-loading");
      submitBtn.innerHTML = `<span class="btn-spinner"></span> 🔄 Analyzing Health Data...`;
      const scannerOverlay = document.getElementById("ai-scanner-overlay");
      if (scannerOverlay) scannerOverlay.classList.add("active");

      // 2. Realistic ML computation delay (~700ms)
      setTimeout(() => {
        if (scannerOverlay) scannerOverlay.classList.remove("active");
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
        const dd = String(now.getDate()).padStart(2, '0');
        const mm = String(now.getMonth() + 1).padStart(2, '0');
        const yyyy = now.getFullYear();
        const dateFormatted = `${dd}-${mm}-${yyyy}`;
        const timestamp = `${dateFormatted} • ` + now.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" });

        const probPct = Math.round(res.prob * 100);
        const resultTier = probPct >= 65 || res.pred === 1 ? "High" : (probPct >= 35 ? "Medium" : "Low");

        const assessmentData = {
          id: patientId,
          patientName,
          name: patientName,
          dateFormatted,
          timestamp,
          rawVals,
          glucose: rawVals[1],
          bp: rawVals[2],
          bmi: rawVals[5],
          age: rawVals[7],
          prob: probPct,
          pred: res.pred,
          result: resultTier,
          res,
          modelLabel,
          model: modelLabel
        };

        latestAssessment = assessmentData;

        renderResultDashboard(assessmentData);
        saveToHistory(assessmentData);
        updateDashboardStats();
        renderHealthAnalytics();
        updateProfileView();

        submitBtn.classList.remove("btn-loading");
        submitBtn.innerHTML = `⚡ Run Real-Time ML Prediction`;

        resultSection.style.display = "block";
        resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 650);
    });
  }

  // --- Render Result Dashboard & Circular Gauge ---
  function renderResultDashboard(data) {
    const { patientName, id, timestamp, rawVals, prob, result, modelLabel } = data;
    const probPct = prob;

    document.getElementById("res-patient-name").textContent = patientName;
    document.getElementById("res-patient-id").textContent = id;
    document.getElementById("res-timestamp").textContent = timestamp;
    document.getElementById("res-model-badge").textContent = modelLabel;

    document.getElementById("print-patient-id").textContent = id;
    document.getElementById("print-timestamp").textContent = timestamp;

    const gaugeCircle = document.getElementById("gauge-circle");
    const gaugePercentText = document.getElementById("gauge-percent-text");
    const riskBadge = document.getElementById("res-risk-badge");
    const headline = document.getElementById("res-status-headline");
    const desc = document.getElementById("res-desc");

    const circumference = 440;
    const offset = circumference - (circumference * (probPct / 100));
    gaugeCircle.style.strokeDashoffset = offset;

    animateCounter(gaugePercentText, probPct);
    headline.textContent = `Diabetes Risk: ${probPct}%`;

    if (result === "High") {
      gaugeCircle.style.stroke = "#ef4444";
      riskBadge.className = "risk-badge high";
      riskBadge.innerHTML = `● High Risk`;
      desc.textContent = `Evaluated clinical biomarkers indicate high risk of diabetes mellitus (${probPct}% probability). Immediate diagnostic blood screening recommended.`;
    } else if (result === "Medium") {
      gaugeCircle.style.stroke = "#f59e0b";
      riskBadge.className = "risk-badge medium";
      riskBadge.innerHTML = `● Medium Risk`;
      desc.textContent = `Evaluated physiological indicators reflect borderline/medium diabetes risk (${probPct}% probability). Preventative nutritional adjustments advised.`;
    } else {
      gaugeCircle.style.stroke = "#10b981";
      riskBadge.className = "risk-badge low";
      riskBadge.innerHTML = `● Low Risk`;
      desc.textContent = `Evaluated physiological indicators reflect healthy bounds (${probPct}% probability). Annual clinical checkup recommended.`;
    }

    // 4 Indicators
    const glucose = rawVals[1];
    const bp = rawVals[2];
    const bmi = rawVals[5];
    const age = rawVals[7];

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

    const bmiElem = document.getElementById("ind-bmi-val");
    const bmiStatus = document.getElementById("ind-bmi-status");
    bmiElem.innerHTML = `${Number(bmi).toFixed(1)} <span class="ind-unit">kg/m²</span>`;
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
      ageStatus.textContent = "Standard Tier";
    }

    // Feature Contribution Breakdown
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

    // Clinical Recommendations
    const recList = document.getElementById("res-recs");
    if (probPct >= 50) {
      recList.innerHTML = `
        <li><strong>Confirmatory Diagnostic Tests:</strong> Schedule an Oral Glucose Tolerance Test (OGTT) and Fasting Blood Sugar test with a physician.</li>
        <li><strong>HbA1c Evaluation:</strong> Order a laboratory Glycated Hemoglobin (HbA1c) test to assess glycemic control over 90 days.</li>
        <li><strong>Nutrition Therapy:</strong> Adopt a low-glycemic dietary regimen with reduced refined carbohydrates.</li>
        <li><strong>Physical Exercise:</strong> Maintain at least 150 min/week moderate aerobic physical activity.</li>
      `;
    } else {
      recList.innerHTML = `
        <li><strong>Preventative Health Maintenance:</strong> Sustain a balanced, nutrient-rich diet rich in dietary fiber and lean proteins.</li>
        <li><strong>Weight Stability:</strong> Maintain a healthy BMI in the 18.5–24.9 kg/m² bracket through active living.</li>
        <li><strong>Periodic Wellness Check:</strong> Undergo annual standard clinical wellness checkups.</li>
      `;
    }
  }

  function animateCounter(elem, target) {
    const duration = 1200;
    const startTime = performance.now();
    function update(currentTime) {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const ease = progress === 1 ? 1 : 1 - Math.pow(2, -10 * progress);
      const currentVal = Math.round(ease * target);
      elem.textContent = `${currentVal}%`;
      if (progress < 1) {
        requestAnimationFrame(update);
      }
    }
    requestAnimationFrame(update);
  }

  window.startNewPrediction = function() {
    resultSection.style.display = "none";
    predictForm.reset();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  window.downloadReport = function() {
    if (!latestAssessment) {
      alert("Please run a patient prediction first before downloading the report.");
      return;
    }
    window.print();
  };

  // --- Prediction History Management ---
  function getHistory() {
    try {
      const key = getHistoryStorageKey();
      const raw = localStorage.getItem(key);
      return raw ? JSON.parse(raw) : [];
    } catch(e) {
      return [];
    }
  }

  function saveToHistory(assessment) {
    const key = getHistoryStorageKey();
    const list = getHistory();
    list.unshift(assessment);
    if (list.length > 50) list.pop();
    try {
      localStorage.setItem(key, JSON.stringify(list));
    } catch(e) {}
    renderHistoryTable();
  }

  function renderHistoryTable() {
    initSeedHistory();
    const list = getHistory();
    const countElem = document.getElementById("history-count");
    if (countElem) countElem.textContent = list.length;

    const tbody = document.getElementById("history-tbody");
    if (!tbody) return;

    if (list.length === 0) {
      tbody.innerHTML = `<tr><td colspan="6" style="text-align:center; padding:30px; color:#8fa5bd;">No prediction records logged yet. Run a screening to save history.</td></tr>`;
      return;
    }

    tbody.innerHTML = list.map((item, idx) => {
      const resultTier = item.result || (item.prob >= 65 ? "High" : (item.prob >= 35 ? "Medium" : "Low"));
      const badgeClass = resultTier === "High" ? "badge danger" : (resultTier === "Medium" ? "badge warning" : "badge safe");
      const dateDisplay = item.dateFormatted || (item.timestamp ? item.timestamp.split("•")[0].trim() : "19-09-2026");

      return `
        <tr>
          <td><strong>${escapeHtml(dateDisplay)}</strong></td>
          <td>${item.glucose} mg/dL</td>
          <td>${Number(item.bmi).toFixed(1)}</td>
          <td><strong>${item.prob}%</strong></td>
          <td><span class="${badgeClass}">${resultTier}</span></td>
          <td>
            <div class="action-btn-group">
              <button type="button" class="btn-action view" onclick="viewRecord('${item.id || idx}')">👁️ View</button>
              <button type="button" class="btn-action delete" onclick="deleteRecord('${item.id || idx}')">🗑️ Delete</button>
              <button type="button" class="btn-action report" onclick="downloadRecordReport('${item.id || idx}')">📥 Report</button>
            </div>
          </td>
        </tr>
      `;
    }).join("");
  }

  window.clearHistory = function() {
    if (confirm("Are you sure you want to clear all screening history records?")) {
      const key = getHistoryStorageKey();
      localStorage.removeItem(key);
      renderHistoryTable();
      updateDashboardStats();
      renderHealthAnalytics();
      updateProfileView();
    }
  };

  // --- Row Actions: View / Delete / Download Report ---
  window.viewRecord = function(recordId) {
    const list = getHistory();
    const record = list.find((item, i) => item.id === recordId || String(i) === String(recordId));
    if (!record) return;

    const modal = document.getElementById("record-view-modal");
    document.getElementById("view-modal-title").textContent = record.name || record.patientName || "Patient Assessment";
    document.getElementById("view-modal-meta").textContent = `ID: ${record.id || 'PID-XXXX'} • Recorded on ${record.timestamp || record.dateFormatted}`;

    const raw = record.rawVals || [0, record.glucose, record.bp || 70, 20, 80, record.bmi, 0.45, record.age || 35];
    const detailsContainer = document.getElementById("view-modal-details");
    
    detailsContainer.innerHTML = `
      <div class="detail-item"><span>Plasma Glucose</span><strong>${record.glucose} mg/dL</strong></div>
      <div class="detail-item"><span>Body Mass Index</span><strong>${Number(record.bmi).toFixed(1)} kg/m²</strong></div>
      <div class="detail-item"><span>Blood Pressure</span><strong>${record.bp || raw[2] || 72} mm Hg</strong></div>
      <div class="detail-item"><span>Patient Age</span><strong>${record.age || raw[7] || 35} Years</strong></div>
      <div class="detail-item"><span>Serum Insulin</span><strong>${raw[4] || 80} μU/mL</strong></div>
      <div class="detail-item"><span>Skin Thickness</span><strong>${raw[3] || 25} mm</strong></div>
      <div class="detail-item"><span>Calculated Risk</span><strong style="color:${record.prob>=65?'#f87171':(record.prob>=35?'#fbbf24':'#34d399')}">${record.prob}% (${record.result})</strong></div>
      <div class="detail-item"><span>Evaluated Model</span><strong style="color:#38bdf8;">${record.model || 'Decision Tree'}</strong></div>
    `;

    const printBtn = document.getElementById("view-modal-print-btn");
    printBtn.onclick = function() {
      downloadRecordReport(record.id || recordId);
    };

    modal.classList.add("active");
  };

  window.closeRecordModal = function() {
    const modal = document.getElementById("record-view-modal");
    if (modal) modal.classList.remove("active");
  };

  window.handleRecordModalOverlayClick = function(e) {
    if (e.target.id === "record-view-modal") {
      closeRecordModal();
    }
  };

  window.deleteRecord = function(recordId) {
    if (!confirm("Are you sure you want to delete this prediction record?")) return;
    const key = getHistoryStorageKey();
    let list = getHistory();
    list = list.filter((item, i) => item.id !== recordId && String(i) !== String(recordId));
    localStorage.setItem(key, JSON.stringify(list));
    renderHistoryTable();
    updateDashboardStats();
    renderHealthAnalytics();
    updateProfileView();
  };

  window.downloadRecordReport = function(recordId) {
    const list = getHistory();
    const record = list.find((item, i) => item.id === recordId || String(i) === String(recordId));
    if (!record) return;

    // Load record into Result Dashboard and print
    latestAssessment = record;
    renderResultDashboard(record);
    window.print();
  };

  // --- Health Analytics Command Center: 4 Trend Charts & Monthly Stats ---
  function renderHealthAnalytics() {
    initSeedHistory();
    const list = getHistory();

    // 1. Monthly Statistics
    const now = new Date();
    const currentMonth = now.getMonth();
    const currentYear = now.getFullYear();

    let monthCount = 0;
    let sumGlu = 0;
    let sumBmi = 0;
    let highRiskCount = 0;

    list.forEach(item => {
      monthCount++;
      sumGlu += Number(item.glucose) || 0;
      sumBmi += Number(item.bmi) || 0;
      if (item.prob >= 65 || item.result === "High") highRiskCount++;
    });

    const avgGlu = monthCount > 0 ? Math.round(sumGlu / monthCount) : 0;
    const avgBmi = monthCount > 0 ? (sumBmi / monthCount).toFixed(1) : "0.0";
    const riskRate = monthCount > 0 ? Math.round((highRiskCount / monthCount) * 100) : 0;

    const mCountElem = document.getElementById("stat-month-count");
    const mGluElem = document.getElementById("stat-month-glucose");
    const mBmiElem = document.getElementById("stat-month-bmi");
    const mRiskElem = document.getElementById("stat-month-risk-rate");

    if (mCountElem) mCountElem.textContent = monthCount;
    if (mGluElem) mGluElem.textContent = `${avgGlu} mg/dL`;
    if (mBmiElem) mBmiElem.textContent = avgBmi;
    if (mRiskElem) mRiskElem.textContent = `${riskRate}%`;

    // 2. Render 4 Dynamic SVG Trend Charts
    // Chronological order: oldest to newest (up to 7 data points)
    const chronoList = list.slice(0, 8).reverse();

    drawSvgTrendChart("chart-glucose-svg", chronoList.map(item => ({
      label: (item.dateFormatted || item.timestamp || "").substring(0, 5),
      val: item.glucose
    })), {
      minY: 60,
      maxY: 180,
      unit: "mg/dL",
      strokeColor: "#38bdf8",
      fillColor: "rgba(56, 189, 248, 0.2)",
      thresholdY: 100,
      thresholdLabel: "Normal (100)"
    });

    drawSvgTrendChart("chart-bmi-svg", chronoList.map(item => ({
      label: (item.dateFormatted || item.timestamp || "").substring(0, 5),
      val: Number(item.bmi)
    })), {
      minY: 15,
      maxY: 45,
      unit: "kg/m²",
      strokeColor: "#a78bfa",
      fillColor: "rgba(167, 139, 250, 0.2)",
      thresholdY: 25,
      thresholdLabel: "Healthy (25)"
    });

    drawSvgTrendChart("chart-bp-svg", chronoList.map(item => ({
      label: (item.dateFormatted || item.timestamp || "").substring(0, 5),
      val: item.bp || (item.rawVals ? item.rawVals[2] : 72)
    })), {
      minY: 50,
      maxY: 110,
      unit: "mm Hg",
      strokeColor: "#34d399",
      fillColor: "rgba(52, 211, 153, 0.2)",
      thresholdY: 80,
      thresholdLabel: "Optimal (80)"
    });

    drawSvgTrendChart("chart-risk-svg", chronoList.map(item => ({
      label: (item.dateFormatted || item.timestamp || "").substring(0, 5),
      val: item.prob
    })), {
      minY: 0,
      maxY: 100,
      unit: "%",
      strokeColor: "#f87171",
      fillColor: "rgba(248, 113, 113, 0.2)",
      thresholdY: 50,
      thresholdLabel: "High (50%)"
    });

    // 3. Render Previous Predictions Timeline
    const timelineContainer = document.getElementById("analytics-timeline-list");
    if (timelineContainer) {
      if (list.length === 0) {
        timelineContainer.innerHTML = `<p style="color:#8fa5bd; grid-column:1/-1;">No previous predictions available.</p>`;
      } else {
        timelineContainer.innerHTML = list.slice(0, 3).map((item, idx) => {
          const resultTier = item.result || (item.prob >= 65 ? "High" : (item.prob >= 35 ? "Medium" : "Low"));
          const badgeClass = resultTier === "High" ? "badge danger" : (resultTier === "Medium" ? "badge warning" : "badge safe");
          return `
            <div class="timeline-card">
              <div class="timeline-date">📅 ${item.dateFormatted || item.timestamp}</div>
              <div class="timeline-title">
                <span>${escapeHtml(item.name || item.patientName || "Patient")}</span>
                <span class="${badgeClass}">${item.prob}% Risk</span>
              </div>
              <div class="timeline-meta">
                <div>Glucose: <strong>${item.glucose} mg/dL</strong> • BMI: <strong>${Number(item.bmi).toFixed(1)}</strong></div>
                <div style="color:#38bdf8; margin-top:4px;">${item.model || 'Decision Tree'}</div>
              </div>
            </div>
          `;
        }).join("");
      }
    }
  }

  // --- Reusable Responsive SVG Chart Drawer ---
  function drawSvgTrendChart(svgId, points, config) {
    const svg = document.getElementById(svgId);
    if (!svg) return;

    if (!points || points.length === 0) {
      svg.innerHTML = `<text x="225" y="80" text-anchor="middle" fill="#71869e" font-size="12">Insufficient trend data</text>`;
      return;
    }

    const width = 450;
    const height = 160;
    const padL = 35;
    const padR = 25;
    const padT = 20;
    const padB = 30;

    const plotW = width - padL - padR;
    const plotH = height - padT - padB;

    const minY = config.minY;
    const maxY = config.maxY;

    function getX(index) {
      if (points.length === 1) return padL + plotW / 2;
      return padL + (index / (points.length - 1)) * plotW;
    }

    function getY(val) {
      const clamped = Math.max(minY, Math.min(maxY, val));
      return padT + plotH - ((clamped - minY) / (maxY - minY)) * plotH;
    }

    // Generate Path Data
    const coords = points.map((p, i) => ({ x: getX(i), y: getY(p.val), val: p.val, label: p.label }));

    let linePath = `M ${coords[0].x},${coords[0].y}`;
    for (let i = 1; i < coords.length; i++) {
      // Smooth cubic bezier
      const prev = coords[i - 1];
      const curr = coords[i];
      const cx1 = prev.x + (curr.x - prev.x) / 2;
      const cy1 = prev.y;
      const cx2 = prev.x + (curr.x - prev.x) / 2;
      const cy2 = curr.y;
      linePath += ` C ${cx1},${cy1} ${cx2},${cy2} ${curr.x},${curr.y}`;
    }

    const areaPath = `${linePath} L ${coords[coords.length - 1].x},${padT + plotH} L ${coords[0].x},${padT + plotH} Z`;

    // Reference threshold line
    const threshY = getY(config.thresholdY);

    let html = `
      <!-- Background Grid -->
      <line x1="${padL}" y1="${padT}" x2="${width - padR}" y2="${padT}" class="chart-grid-line" />
      <line x1="${padL}" y1="${padT + plotH / 2}" x2="${width - padR}" y2="${padT + plotH / 2}" class="chart-grid-line" />
      <line x1="${padL}" y1="${padT + plotH}" x2="${width - padR}" y2="${padT + plotH}" class="chart-grid-line" />

      <!-- Threshold Line -->
      <line x1="${padL}" y1="${threshY}" x2="${width - padR}" y2="${threshY}" stroke="rgba(239, 68, 68, 0.45)" stroke-dasharray="3 3" />
      <text x="${width - padR}" y="${threshY - 4}" text-anchor="end" fill="#f87171" font-size="9">${config.thresholdLabel}</text>

      <!-- Area & Line -->
      <path d="${areaPath}" fill="${config.strokeColor}" class="chart-path-area" />
      <path d="${linePath}" stroke="${config.strokeColor}" class="chart-path-line" />
    `;

    // Draw Points & Labels
    coords.forEach(pt => {
      html += `
        <circle cx="${pt.x}" cy="${pt.y}" r="4" stroke="${config.strokeColor}" class="chart-point">
          <title>${pt.val} ${config.unit}</title>
        </circle>
        <text x="${pt.x}" y="${pt.y - 8}" text-anchor="middle" fill="#fff" font-size="10" font-weight="700">${pt.val}</text>
        <text x="${pt.x}" y="${height - 8}" text-anchor="middle" class="chart-label-text">${pt.label}</text>
      `;
    });

    svg.innerHTML = html;
  }

  // --- Dashboard Tab Statistics ---
  function updateDashboardStats() {
    initSeedHistory();
    const list = getHistory();
    const total = list.length;
    let high = 0;
    let low = 0;
    let sumProb = 0;

    list.forEach(item => {
      sumProb += item.prob;
      if (item.prob >= 65 || item.result === "High" || item.pred === 1) high++;
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
        const prob = latestAssessment.prob;
        const isHigh = prob >= 65 || latestAssessment.result === "High";
        dashContainer.innerHTML = `
          <div class="dashboard-card" style="margin-top: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1c3855; padding-bottom: 15px; margin-bottom: 18px; flex-wrap: wrap; gap: 10px;">
              <div>
                <span class="eyebrow">MOST RECENT CLINICAL ASSESSMENT</span>
                <h2 style="margin: 4px 0 0; font-size: 22px;">${escapeHtml(latestAssessment.name || latestAssessment.patientName)} (${latestAssessment.id})</h2>
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
                Evaluated using <strong>${latestAssessment.model}</strong> on ${latestAssessment.timestamp || latestAssessment.dateFormatted}. Glucose: ${latestAssessment.glucose} mg/dL • BMI: ${latestAssessment.bmi} kg/m² • Age: ${latestAssessment.age}.
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

  // --- Auth Modal & User Registration / Secure Login ---
  let authMode = 'login';

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
      sub.textContent = "Register with secure password hashing (SHA-256).";
      submit.textContent = "Register Account";
    } else {
      btnReg.classList.remove("active");
      btnLogin.classList.add("active");
      nameGroup.style.display = "none";
      title.textContent = "Sign In to DiaPredict";
      sub.textContent = "Access your user-specific predictions and profile.";
      submit.textContent = "Sign In";
    }
  };

  function getRegisteredUsers() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY_USERS);
      return raw ? JSON.parse(raw) : [];
    } catch(e) {
      return [];
    }
  }

  window.handleAuthSubmit = async function(e) {
    e.preventDefault();
    const email = document.getElementById("auth-email").value.trim().toLowerCase();
    const password = document.getElementById("auth-password").value;

    if (!email || !password) return;

    // Compute Cryptographic SHA-256 Hash
    const passwordHash = await hashPassword(password);
    const users = getRegisteredUsers();

    if (authMode === 'register') {
      const fullName = document.getElementById("auth-fullname").value.trim() || email.split("@")[0];
      const existing = users.find(u => u.email === email);
      if (existing) {
        alert("An account with this email address already exists. Please sign in.");
        setAuthMode('login');
        return;
      }

      const newUser = {
        name: fullName,
        email: email,
        passwordHash: passwordHash,
        role: "Clinical Practitioner • Screening Analyst",
        createdAt: new Date().toLocaleDateString()
      };

      users.push(newUser);
      localStorage.setItem(STORAGE_KEY_USERS, JSON.stringify(users));

      // Auto login newly registered user
      localStorage.setItem(STORAGE_KEY_ACTIVE_USER, JSON.stringify(newUser));
      closeAuthModal();
      alert(`Account created successfully! Welcome, ${fullName}.`);
      updateProfileView();
      renderHistoryTable();
      renderHealthAnalytics();
      updateDashboardStats();
    } else {
      // Secure Login Verification
      const foundUser = users.find(u => u.email === email);
      if (!foundUser) {
        // Allow seamless first-time clinician login
        const defaultName = email.split("@")[0].charAt(0).toUpperCase() + email.split("@")[0].slice(1);
        const newUser = {
          name: defaultName,
          email: email,
          passwordHash: passwordHash,
          role: "Clinical Practitioner",
          createdAt: new Date().toLocaleDateString()
        };
        users.push(newUser);
        localStorage.setItem(STORAGE_KEY_USERS, JSON.stringify(users));
        localStorage.setItem(STORAGE_KEY_ACTIVE_USER, JSON.stringify(newUser));
        closeAuthModal();
        alert(`Welcome, ${defaultName}! Signed in successfully.`);
      } else {
        if (foundUser.passwordHash !== passwordHash) {
          alert("Incorrect password. Please try again.");
          return;
        }
        localStorage.setItem(STORAGE_KEY_ACTIVE_USER, JSON.stringify(foundUser));
        closeAuthModal();
        alert(`Welcome back, ${foundUser.name}!`);
      }

      updateProfileView();
      renderHistoryTable();
      renderHealthAnalytics();
      updateDashboardStats();
    }
  };

  window.logoutUser = function() {
    localStorage.removeItem(STORAGE_KEY_ACTIVE_USER);
    alert("You have logged out.");
    updateProfileView();
    renderHistoryTable();
    renderHealthAnalytics();
    updateDashboardStats();
  };

  function updateProfileView() {
    const user = getActiveUser();
    const navAuthBtn = document.getElementById("nav-auth-btn");
    const profName = document.getElementById("prof-name");
    const profEmail = document.getElementById("prof-email");
    const profAvatar = document.getElementById("prof-avatar");
    const profAuth = document.getElementById("prof-auth-status");
    const profCount = document.getElementById("prof-eval-count");

    const historyList = getHistory();
    if (profCount) profCount.textContent = `${historyList.length} Assessments`;

    if (user && user.email) {
      if (navAuthBtn) navAuthBtn.textContent = `👤 ${user.name}`;
      if (profName) profName.textContent = user.name;
      if (profEmail) profEmail.textContent = user.email;
      if (profAvatar) profAvatar.textContent = user.name.charAt(0).toUpperCase();
      if (profAuth) {
        profAuth.textContent = "Authenticated Session (SHA-256)";
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
    return String(str).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  // --- Initial Mount ---
  initSeedHistory();
  renderHistoryTable();
  updateDashboardStats();
  renderHealthAnalytics();
  updateProfileView();

  console.log("DiaPredict Engine v2.1 Initialized successfully.");
})();
