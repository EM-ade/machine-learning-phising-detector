(function () {
  "use strict";

  const emailInput = document.getElementById("email-text");
  const charCount = document.getElementById("char-count");
  const btnAnalyze = document.getElementById("btn-analyze");
  const btnClear = document.getElementById("btn-clear");
  const btnNew = document.getElementById("btn-new");
  const btnRefresh = document.getElementById("btn-refresh");
  const loading = document.getElementById("loading");
  const resultSection = document.getElementById("result");
  const resultCard = resultSection?.querySelector(".result-card");
  const resultLabel = document.getElementById("result-label");
  const resultSub = document.getElementById("result-sub");
  const confidenceBar = document.getElementById("confidence-bar");
  const confidenceValue = document.getElementById("confidence-value");
  const indicatorsList = document.getElementById("indicators-list");
  const processingTime = document.getElementById("processing-time");
  const historyBody = document.getElementById("history-body");

  const MAX_LENGTH = 50000;

  emailInput.addEventListener("input", function () {
    const len = this.value.length;
    charCount.textContent = len;
    btnAnalyze.disabled = len === 0;
    if (len > MAX_LENGTH) {
      this.value = this.value.substring(0, MAX_LENGTH);
      charCount.textContent = MAX_LENGTH;
    }
  });

  btnClear.addEventListener("click", function () {
    emailInput.value = "";
    charCount.textContent = "0";
    btnAnalyze.disabled = true;
    resultSection.classList.add("hidden");
  });

  emailInput.addEventListener("keydown", function (e) {
    if (e.ctrlKey && e.key === "Enter") {
      e.preventDefault();
      if (!btnAnalyze.disabled) analyze();
    }
  });

  btnAnalyze.addEventListener("click", analyze);

  async function analyze() {
    const text = emailInput.value.trim();
    if (!text) return;

    btnAnalyze.disabled = true;
    resultSection.classList.add("hidden");
    loading.classList.remove("hidden");

    try {
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email_text: text }),
      });

      const data = await res.json();

      if (!res.ok) {
        showError(data.error || "An error occurred");
        return;
      }

      showResult(data);
      fetchHistory();
    } catch (err) {
      showError("Network error. Is the server running?");
    } finally {
      loading.classList.add("hidden");
      btnAnalyze.disabled = false;
    }
  }

  function showResult(data) {
    const isPhishing = data.is_phishing;
    resultCard.className = "result-card " + (isPhishing ? "phishing" : "legitimate");
    resultLabel.textContent = isPhishing ? "PHISHING" : "LEGITIMATE";

    let sub = isPhishing
      ? "This email contains suspicious patterns"
      : "This email appears to be legitimate";
    if (data.low_confidence) {
      sub += " — Low confidence prediction";
    }
    if (data.rule_override) {
      sub += " (flagged by rule check)";
    }
    resultSub.textContent = sub;

    const conf = data.confidence;
    confidenceBar.style.width = conf + "%";
    confidenceValue.textContent = conf + "%";

    indicatorsList.innerHTML = "";
    (data.indicators || []).forEach(function (ind) {
      const li = document.createElement("li");
      li.textContent = ind;
      indicatorsList.appendChild(li);
    });

    processingTime.textContent = data.processing_time_ms;
    resultSection.classList.remove("hidden");
  }

  function showError(msg) {
    resultCard.className = "result-card";
    resultLabel.textContent = "ERROR";
    resultSub.textContent = msg;
    confidenceBar.style.width = "0%";
    confidenceValue.textContent = "—";
    indicatorsList.innerHTML = "";
    processingTime.textContent = "—";
    resultSection.classList.remove("hidden");
  }

  btnNew.addEventListener("click", function () {
    emailInput.value = "";
    charCount.textContent = "0";
    btnAnalyze.disabled = true;
    resultSection.classList.add("hidden");
    emailInput.focus();
  });

  btnRefresh.addEventListener("click", fetchHistory);

  async function fetchHistory() {
    try {
      const res = await fetch("/history?limit=20");
      const rows = await res.json();
      renderHistory(rows);
    } catch (_) {}
  }

  function renderHistory(rows) {
    if (!rows || rows.length === 0) {
      historyBody.innerHTML = '<tr><td colspan="4" class="empty-msg">No analyses yet</td></tr>';
      return;
    }

    historyBody.innerHTML = "";
    rows.forEach(function (r) {
      const tr = document.createElement("tr");
      const tagClass = r.is_phishing ? "tag-phishing" : "tag-legitimate";
      const tagLabel = r.is_phishing ? "Phishing" : "Legitimate";

      tr.innerHTML =
        "<td>" +
        (r.timestamp || "—") +
        "</td><td><span class='tag " +
        tagClass +
        "'>" +
        tagLabel +
        "</span></td><td>" +
        r.confidence +
        "%</td><td>" +
        r.processing_time_ms +
        "</td>";

      historyBody.appendChild(tr);
    });
  }

  fetchHistory();

  // Metrics toggle
  var btnMetrics = document.getElementById("btn-metrics");
  var btnCloseMetrics = document.getElementById("btn-close-metrics");
  var metricsSection = document.getElementById("metrics-section");
  var metricsLoading = document.getElementById("metrics-loading");
  var metricsFetched = false;

  if (btnMetrics) {
    btnMetrics.addEventListener("click", function () {
      if (metricsSection.classList.contains("hidden")) {
        metricsSection.classList.remove("hidden");
        btnMetrics.textContent = "Hide Model Metrics";
        if (!metricsFetched) loadMetrics();
      } else {
        metricsSection.classList.add("hidden");
        btnMetrics.textContent = "View Model Metrics";
      }
    });
  }

  if (btnCloseMetrics) {
    btnCloseMetrics.addEventListener("click", function () {
      metricsSection.classList.add("hidden");
      btnMetrics.textContent = "View Model Metrics";
    });
  }

  async function loadMetrics() {
    metricsLoading.classList.remove("hidden");
    try {
      var res = await fetch("/evaluation");
      if (!res.ok) {
        document.getElementById("m-accuracy").textContent = "N/A";
        return;
      }
      var data = await res.json();
      var cr = data.classification_report || {};

      document.getElementById("m-accuracy").textContent = (cr.accuracy * 100).toFixed(1) + "%";
      document.getElementById("m-precision").textContent = (cr.macro_avg ? cr.macro_avg.precision : 0).toFixed(2);
      document.getElementById("m-recall").textContent = (cr.macro_avg ? cr.macro_avg.recall : 0).toFixed(2);
      document.getElementById("m-f1").textContent = (cr.macro_avg ? cr.macro_avg["f1"] : 0).toFixed(2);
      document.getElementById("m-auc").textContent = data.roc_auc ? data.roc_auc.toFixed(4) : "—";

      metricsFetched = true;
    } catch (e) {
      document.getElementById("m-accuracy").textContent = "Error";
    } finally {
      metricsLoading.classList.add("hidden");
    }
  }
})();
