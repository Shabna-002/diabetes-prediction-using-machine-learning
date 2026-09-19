// DiaPredict - Client-side ML Inference & UI Engine
(function() {
  const models = window.DIABETES_ML_MODELS;
  if (!models) {
    console.error("Model data not found.");
    return;
  }

  // --- Tab Navigation ---
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
  };

  navLinks.forEach(link => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      switchTab(link.dataset.tab);
    });
  });

  // --- Quick Sample Fill ---
  const samples = {
    normal: {
      pregnancies: 1,
      glucose: 85,
      bloodPressure: 66,
      skinThickness: 29,
      insulin: 90,
      bmi: 23.8,
      pedigree: 0.254,
      age: 26,
      name: "Patient Sarah (Healthy Baseline)"
    },
    risk: {
      pregnancies: 6,
      glucose: 148,
      bloodPressure: 72,
      skinThickness: 35,
      insulin: 0,
      bmi: 33.6,
      pedigree: 0.627,
      age: 50,
      name: "Patient Eleanor (High Risk Case)"
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

  // --- ML Inference Helpers ---
  function preprocess(inputVals) {
    // inputVals: array of 8 numbers
    // Zero imputation on biological zeros: [1, 2, 3, 4, 5]
    const zeroIndices = [1, 2, 3, 4, 5];
    const imputed = inputVals.map((v, i) => {
      if (zeroIndices.includes(i) && (v === 0 || isNaN(v))) {
        return models.imputer_stats[i];
      }
      return isNaN(v) ? models.imputer_stats[i] : v;
    });

    // Standard scaler: (x - mean) / scale
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

  // --- Predict Form Submit ---
  const predictForm = document.getElementById("predict-form");
  const resultCard = document.getElementById("result-card");

  if (predictForm) {
    predictForm.addEventListener("submit", function(e) {
      e.preventDefault();

      const name = document.getElementById("patient_name").value.trim() || "Anonymous Patient";
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

      displayResult(name, rawVals, res, modelLabel);
      saveToHistory(name, rawVals, res, modelLabel);
    });
  }

  function displayResult(name, rawVals, res, modelLabel) {
    const isPositive = res.pred === 1;
    const probPct = (res.prob * 100).toFixed(1);

    const iconElem = document.getElementById("res-icon");
    const statusElem = document.getElementById("res-status");
    const descElem = document.getElementById("res-desc");
    const probPctElem = document.getElementById("res-prob-pct");
    const probBarElem = document.getElementById("res-prob-bar");
    const recListElem = document.getElementById("res-recs");
    const modelBadgeElem = document.getElementById("res-model-badge");

    modelBadgeElem.textContent = modelLabel;
    probPctElem.textContent = `${probPct}%`;
    probBarElem.style.width = `${probPct}%`;

    if (isPositive) {
      iconElem.textContent = "⚠";
      iconElem.style.background = "rgba(239, 68, 68, 0.2)";
      iconElem.style.color = "#f87171";
      statusElem.textContent = "High Risk of Diabetes (Positive)";
      statusElem.style.color = "#f87171";
      descElem.textContent = `Based on the evaluated indicators, this assessment shows an elevated statistical likelihood of diabetes mellitus (${probPct}% probability).`;
      probBarElem.style.background = "linear-gradient(90deg, #f59e0b, #ef4444)";

      recListElem.innerHTML = `
        <li><strong>Confirmatory Diagnostics:</strong> Schedule an Oral Glucose Tolerance Test (OGTT) and Fasting Plasma Glucose (FPG) test with a physician.</li>
        <li><strong>HbA1c Evaluation:</strong> Order a laboratory Glycated Hemoglobin (HbA1c) test to assess glycemic control over the past 3 months.</li>
        <li><strong>Dietary & Lifestyle:</strong> Consult a clinical dietitian for low-glycemic dietary planning and initiate 150 min/week moderate physical exercise.</li>
        <li><strong>Monitoring:</strong> Regular self-monitoring of blood glucose (SMBG) if indicated by clinician.</li>
      `;
    } else {
      iconElem.textContent = "✓";
      iconElem.style.background = "rgba(16, 185, 129, 0.2)";
      iconElem.style.color = "#34d399";
      statusElem.textContent = "Low Risk of Diabetes (Negative)";
      statusElem.style.color = "#34d399";
      descElem.textContent = `The evaluated indicators reflect low statistical indicators of diabetes mellitus (${probPct}% probability). Biomarkers fall within normal bounds.`;
      probBarElem.style.background = "linear-gradient(90deg, #38bdf8, #10b981)";

      recListElem.innerHTML = `
        <li><strong>Preventative Maintenance:</strong> Maintain a balanced nutrient-dense diet rich in fiber, whole grains, and lean proteins.</li>
        <li><strong>Physical Activity:</strong> Engage in regular aerobic and strength-training physical activities.</li>
        <li><strong>Routine Screening:</strong> Continue annual standard health checkups and blood work.</li>
      `;
    }

    resultCard.style.display = "block";
    resultCard.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  // --- Local Storage History ---
  function getHistory() {
    try {
      return JSON.parse(localStorage.getItem("diabetes_ml_history") || "[]");
    } catch (e) {
      return [];
    }
  }

  function saveToHistory(name, rawVals, res, modelLabel) {
    const history = getHistory();
    const entry = {
      id: Date.now(),
      date: new Date().toLocaleString(),
      name: name,
      model: modelLabel,
      glucose: rawVals[1],
      bmi: rawVals[5],
      age: rawVals[7],
      prediction: res.pred,
      probability: (res.prob * 100).toFixed(1) + "%"
    };
    history.unshift(entry);
    if (history.length > 50) history.pop();
    localStorage.setItem("diabetes_ml_history", JSON.stringify(history));
    renderHistoryTable();
  }

  function renderHistoryTable() {
    const history = getHistory();
    const tbody = document.getElementById("history-tbody");
    const countBadge = document.getElementById("history-count");
    if (!tbody) return;

    if (countBadge) countBadge.textContent = history.length;

    if (history.length === 0) {
      tbody.innerHTML = `<tr><td colspan="7" style="text-align:center; padding:30px; color:#8fa5bd;">No prediction records yet. Run a prediction to see history here!</td></tr>`;
      return;
    }

    tbody.innerHTML = history.map(item => `
      <tr>
        <td>${item.date}</td>
        <td><strong>${item.name}</strong></td>
        <td>${item.model}</td>
        <td>${item.glucose} mg/dL</td>
        <td>${item.bmi}</td>
        <td><strong>${item.probability}</strong></td>
        <td>
          <span class="badge ${item.prediction === 1 ? 'danger' : 'safe'}">
            ${item.prediction === 1 ? 'Positive' : 'Negative'}
          </span>
        </td>
      </tr>
    `).join("");
  }

  window.clearHistory = function() {
    if (confirm("Are you sure you want to clear all prediction history?")) {
      localStorage.removeItem("diabetes_ml_history");
      renderHistoryTable();
    }
  };

  // Render initial history
  renderHistoryTable();

  console.log("DiaPredict Engine Initialized successfully.");
})();
