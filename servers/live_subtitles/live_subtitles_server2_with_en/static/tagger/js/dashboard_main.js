// ===== Dashboard Page: Main Logic =====
// Only execute if we're on the dashboard page
(function () {
  // Check if dashboard-specific elements exist
  const uploadArea = document.getElementById("uploadArea");
  const fileInput = document.getElementById("fileInput");
  const tagBtn = document.getElementById("tagBtn");
  const speechBtn = document.getElementById("speechBtn");
  const resultsDiv = document.getElementById("results");
  const progressDiv = document.getElementById("progress");
  const progressFill = document.getElementById("progressFill");
  const progressText = document.getElementById("progressText");
  const toast = document.getElementById("toast");

  // If uploadArea doesn't exist, we're not on the dashboard page - exit silently
  if (!uploadArea) {
    return;
  }

  // ===== File Upload Handling =====
  uploadArea.addEventListener("click", () => fileInput.click());
  uploadArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadArea.classList.add("dragover");
  });
  uploadArea.addEventListener("dragleave", () => {
    uploadArea.classList.remove("dragover");
  });
  uploadArea.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadArea.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  });

  if (fileInput) {
    fileInput.addEventListener("change", (e) => {
      const file = e.target.files[0];
      if (file) handleFile(file);
    });
  }

  function handleFile(file) {
    selectedFile = file;
    uploadArea.querySelector("h3").textContent = `📁 ${file.name}`;
    uploadArea.querySelector("p").textContent =
      `Size: ${(file.size / 1024).toFixed(1)} KB | Type: ${file.type}`;
    if (tagBtn) tagBtn.disabled = false;
    if (speechBtn) speechBtn.disabled = false;
  }

  // ===== Toast notification =====
  function showToast(message, duration = 2000) {
    if (!toast) return;
    toast.textContent = message;
    toast.classList.add("show");
    setTimeout(() => {
      toast.classList.remove("show");
    }, duration);
  }

  // ===== Navigate to segment analytics =====
  function navigateToSegment(segmentDir, segmentNumber) {
    const segmentId =
      segmentDir || `seg_${String(segmentNumber || 0).padStart(3, "0")}`;
    showToast(`🔍 Opening analytics for ${segmentId} in new tab...`);
    window.open(`/tags?segment=${encodeURIComponent(segmentId)}`, "_blank");
  }

  // ===== Load Saved Chunks =====
  async function loadChunks() {
    const chunksLoading = document.getElementById("chunksLoading");
    const filterSelect = document.getElementById("filterSelect");
    const limitSelect = document.getElementById("limitSelect");

    if (chunksLoading) chunksLoading.style.display = "block";
    const speechOnly = filterSelect ? filterSelect.value === "speech" : false;
    const limit = limitSelect ? parseInt(limitSelect.value) : 50;

    try {
      const url = `/tags/chunks?limit=${limit}&offset=0&speech_only=${speechOnly}`;
      const response = await fetch(url);
      const data = await response.json();
      updateStatistics(data.stats);
      updateChunksTable(data.chunks, data.total_entries, data.returned_entries);
      updateTopPredictions(data.stats.top_predictions);
    } catch (error) {
      const container = document.getElementById("chunksTableContainer");
      if (container) {
        container.innerHTML = `<div style="padding:20px;text-align:center;color:#721c24;">❌ Error loading data: ${escapeHtml(error.message)}</div>`;
      }
    } finally {
      if (chunksLoading) chunksLoading.style.display = "none";
    }
  }

  function updateStatistics(stats) {
    if (!stats) return;
    const setText = (id, val) => {
      const el = document.getElementById(id);
      if (el) el.textContent = val;
    };
    setText("totalSegments", stats.total_segments || 0);
    setText("speechSegments", stats.speech_segments || 0);
    setText("speechPercentage", (stats.speech_percentage || 0) + "%");
    setText("avgSpeechProb", (stats.avg_speech_probability || 0).toFixed(3));
  }

  function updateChunksTable(chunks, totalEntries, returnedEntries) {
    const container = document.getElementById("chunksTableContainer");
    if (!container) return;

    if (!chunks || chunks.length === 0) {
      container.innerHTML = `<div style="padding:40px;text-align:center;color:#666;"><p style="font-size:1.2em;">📭 No saved segments found</p><p style="margin-top:10px;">Process some audio segments to see data here.</p></div>`;
      return;
    }

    let html = `<p style="margin-bottom:10px;color:#666;">Showing ${chunks.length} of ${totalEntries} segments <span style="font-size:0.85em;margin-left:10px;">💡 Click a row to view detailed analytics</span></p>
      <table><thead><tr><th>Segment</th><th>#</th><th>Speech</th><th>Probability</th><th>Mode</th><th>Top Predictions</th><th>Timestamp</th></tr></thead><tbody>`;

    chunks.forEach((chunk) => {
      const speechDetected = chunk.speech_detected;
      const speechProb = chunk.speech_probability || 0;
      const predictions = chunk.top_predictions || [];
      const predText = predictions
        .map((p) => `${p.name} (${(p.prob * 100).toFixed(0)}%)`)
        .join(", ");
      const timestamp = chunk.timestamp
        ? new Date(chunk.timestamp).toLocaleString()
        : "N/A";
      const segmentDir = chunk.segment_dir || "N/A";
      const segmentNumber = chunk.segment_number || "-";

      html += `<tr onclick="navigateToSegment('${escapeHtml(String(chunk.segment_dir || ""))}', ${chunk.segment_number || 0})" title="Click to view detailed analytics" style="cursor:pointer;">
        <td><strong>${escapeHtml(segmentDir)}</strong><span class="view-indicator">🔍</span></td>
        <td>${segmentNumber}</td>
        <td><span class="badge ${speechDetected ? "badge-success" : "badge-danger"}">${speechDetected ? "✅ Yes" : "❌ No"}</span></td>
        <td><div class="prob-bar"><div class="prob-fill ${speechDetected ? "speech" : ""}" style="width:${speechProb * 100}%"></div></div><small>${(speechProb * 100).toFixed(1)}%</small></td>
        <td><span class="badge badge-info">${escapeHtml(chunk.processing_mode || "unknown")}</span></td>
        <td><small>${escapeHtml(predText) || "No predictions"}</small></td>
        <td><small>${escapeHtml(timestamp)}</small></td></tr>`;
    });

    html += "</tbody></table>";
    container.innerHTML = html;
  }

  function updateTopPredictions(predictions) {
    const container = document.getElementById("topPredictionsChart");
    if (!container) return;

    if (!predictions || predictions.length === 0) {
      container.innerHTML =
        '<p style="text-align:center;color:#666;">No predictions data available</p>';
      return;
    }

    const maxCount = Math.max(...predictions.map((p) => p.count));
    let html = "";
    predictions.slice(0, 10).forEach((pred) => {
      const pct = ((pred.count / maxCount) * 100).toFixed(0);
      html += `<div class="chart-bar"><div class="chart-label">${escapeHtml(pred.name)}</div><div class="chart-track"><div class="chart-fill" style="width:${pct}%"></div></div><div class="chart-value">${pred.count} (avg ${(pred.avg_probability * 100).toFixed(0)}%)</div></div>`;
    });
    container.innerHTML = html;
  }

  // ===== Fill config and speech classes from API =====
  async function loadConfig() {
    try {
      const resp = await fetch("/tags/config");
      const config = await resp.json();
      const configTable = document.getElementById("configTable");
      if (configTable) {
        configTable.innerHTML = `
          <tr><td>Top K Predictions</td><td><strong>${config.top_k}</strong></td></tr>
          <tr><td>Speech Threshold</td><td><strong>${config.speech_prob_threshold}</strong></td></tr>
          <tr><td>Chunk Duration</td><td><strong>${config.chunk_duration}s</strong></td></tr>
          <tr><td>Chunk Overlap</td><td><strong>${config.chunk_overlap}s</strong></td></tr>
          <tr><td>Min Chunk Duration</td><td><strong>${config.min_chunk_duration}s</strong></td></tr>`;
      }
      const speechList = document.getElementById("speechClassesList");
      if (speechList) {
        const classes = config.speech_classes || SPEECH_CLASS_NAMES;
        speechList.innerHTML = classes
          .map(
            (c) =>
              `<li style="padding:5px 0;border-bottom:1px solid #f0f0f0">• ${c}</li>`,
          )
          .join("");
      }
    } catch (e) {
      console.error("Failed to load config:", e);
    }
  }

  // ===== Tagging and Speech Check =====
  if (tagBtn) {
    tagBtn.addEventListener("click", async () => {
      if (!selectedFile) return;
      const formData = new FormData();
      formData.append("file", selectedFile);
      const isChunked =
        document.getElementById("chunkedMode")?.checked || false;
      const endpoint = isChunked ? "/tags/chunks" : "/tags/audio";
      if (isChunked) {
        formData.append(
          "chunk_duration",
          document.getElementById("chunkDuration")?.value || "2.0",
        );
        formData.append(
          "overlap_duration",
          document.getElementById("overlapDuration")?.value || "1.0",
        );
      }
      showProgress("Processing audio...");
      try {
        const response = await fetch(endpoint, {
          method: "POST",
          body: formData,
        });
        const data = await response.json();
        displayResults(data);
        setTimeout(() => loadChunks(), 500);
      } catch (error) {
        showError(error.message);
      } finally {
        hideProgress();
      }
    });
  }

  if (speechBtn) {
    speechBtn.addEventListener("click", async () => {
      if (!selectedFile) return;
      const formData = new FormData();
      formData.append("file", selectedFile);
      showProgress("Checking speech...");
      try {
        const response = await fetch("/tags/speech-check", {
          method: "POST",
          body: formData,
        });
        const data = await response.json();
        displaySpeechResult(data);
      } catch (error) {
        showError(error.message);
      } finally {
        hideProgress();
      }
    });
  }

  function showProgress(text) {
    if (progressDiv) progressDiv.style.display = "block";
    if (progressText) progressText.textContent = text;
    if (progressFill) progressFill.style.width = "50%";
  }

  function hideProgress() {
    if (progressDiv) progressDiv.style.display = "none";
    if (progressFill) progressFill.style.width = "0%";
  }

  function displayResults(data) {
    if (!resultsDiv) return;
    resultsDiv.style.display = "block";
    let html = "<h2>📊 Results</h2>";
    if (data.mode === "chunked" || data.chunks) {
      html += `<p><strong>Mode:</strong> Chunked | <strong>Chunks:</strong> ${data.total_chunks} | <strong>Duration:</strong> ${data.total_duration_seconds}s</p><h3>Overall Top Predictions:</h3>`;
      data.overall_top_predictions.forEach((pred) => {
        html += `<div class="result-item"><strong>${escapeHtml(pred.name)}</strong><div class="prob-bar"><div class="prob-fill" style="width:${pred.prob * 100}%"></div></div><span>${(pred.prob * 100).toFixed(1)}%</span></div>`;
      });
      if (data.chunks) {
        html +=
          "<h3>Chunk Details:</h3><table><tr><th>Chunk</th><th>Time</th><th>Top Prediction</th><th>Speech</th></tr>";
        data.chunks.forEach((chunk) => {
          const tp = chunk.predictions
            ? chunk.predictions[0] || { name: "N/A", prob: 0 }
            : { name: "N/A", prob: 0 };
          html += `<tr><td>${chunk.chunk_index}</td><td>${chunk.start_time}s-${chunk.end_time}s</td><td>${escapeHtml(tp.name)}(${(tp.prob * 100).toFixed(1)}%)</td><td>${chunk.has_speech ? "✅" : "❌"}</td></tr>`;
        });
        html += "</table>";
      }
    } else {
      html += "<h3>Top Predictions:</h3>";
      data.top_predictions.forEach((pred) => {
        html += `<div class="result-item"><strong>${escapeHtml(pred.name)}</strong><div class="prob-bar"><div class="prob-fill" style="width:${pred.prob * 100}%"></div></div><span>${(pred.prob * 100).toFixed(1)}%</span></div>`;
      });
    }
    html += `<p><strong>Speech Detected:</strong> ${data.speech_detected ? "✅ Yes" : "❌ No"} | <strong>Processing Time:</strong> ${data.processing_time_seconds}s | <strong>RTF:</strong> ${data.real_time_factor}</p>`;
    resultsDiv.innerHTML = html;
  }

  function displaySpeechResult(data) {
    if (!resultsDiv) return;
    resultsDiv.style.display = "block";
    resultsDiv.innerHTML = `<h2>🎤 Speech Detection Result</h2><div class="result-item"><h3>${data.has_speech ? "✅ Speech Detected!" : "❌ No Speech Detected"}</h3><p><strong>Probability:</strong> ${(data.speech_probability * 100).toFixed(1)}%</p><p><strong>Threshold:</strong> ${(data.threshold_used * 100).toFixed(0)}%</p><p><strong>File:</strong> ${escapeHtml(data.filename)}</p><p><strong>Processing Time:</strong> ${data.processing_time_seconds}s</p></div>`;
  }

  function showError(message) {
    if (!resultsDiv) return;
    resultsDiv.style.display = "block";
    resultsDiv.innerHTML = `<div style="background:#fee;border:1px solid #fcc;padding:20px;border-radius:5px;"><h3 style="color:#c00;">❌ Error</h3><p>${escapeHtml(message)}</p></div>`;
  }

  // ===== Initial Load =====
  document.addEventListener("DOMContentLoaded", () => {
    loadConfig();
    loadChunks();
    setInterval(loadChunks, 30000);
  });

  // Expose functions to global scope for onclick handlers
  window.navigateToSegment = navigateToSegment;
  window.loadChunks = loadChunks;
})();
