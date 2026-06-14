// ============================================================
// pairwise_comparison.js
// Pairwise Speaker Comparison with Independent Segment Tabs
// Dependencies: Chart.js 4.4.0+
// Requires global: allData, getColor(), SPEAKER_COLORS
// ============================================================

let pairwiseData = null;
let pairwiseCharts = {};

/**
 * Initialize pairwise speaker selection dropdowns
 * @param {string[]} speakerLabels - Array of speaker label strings
 */
function initPairwiseSelects(speakerLabels) {
  const select1 = document.getElementById("pairwiseSpeaker1");
  const select2 = document.getElementById("pairwiseSpeaker2");

  if (!select1 || !select2) {
    console.log("[Pairwise] Select elements not found in DOM");
    return;
  }

  // Save current selections to restore after rebuild
  const current1 = select1.value;
  const current2 = select2.value;

  // Rebuild both dropdowns
  [select1, select2].forEach((select) => {
    select.innerHTML = '<option value="">-- Select Speaker --</option>';
    speakerLabels.forEach((label, i) => {
      const option = document.createElement("option");
      option.value = label;
      option.textContent = `${label} (${i + 1})`;
      select.appendChild(option);
    });
  });

  // Restore previous selections if still valid
  if (speakerLabels.includes(current1)) select1.value = current1;
  if (speakerLabels.includes(current2)) select2.value = current2;

  // Auto-select first two if nothing selected
  if (!select1.value && speakerLabels.length >= 1) {
    select1.value = speakerLabels[0];
  }
  if (!select2.value && speakerLabels.length >= 2) {
    select2.value = speakerLabels[1];
  }

  // Trigger update if both have values
  if (select1.value && select2.value) {
    updatePairwiseComparison();
  }

  console.log(`[Pairwise] Initialized with ${speakerLabels.length} speakers`);
}

/**
 * Fetch comparison data and render the pairwise view
 */
async function updatePairwiseComparison() {
  const label1 = document.getElementById("pairwiseSpeaker1")?.value;
  const label2 = document.getElementById("pairwiseSpeaker2")?.value;

  if (!label1 || !label2) {
    showEmptyPairwise();
    return;
  }

  if (label1 === label2) {
    const content = document.getElementById("pairwiseContent");
    if (content) {
      content.innerHTML = `
        <div class="empty-comparison">
          <div class="icon">⚠️</div>
          <h3>Same Speaker Selected</h3>
          <p>Please select two different speakers to compare.</p>
        </div>`;
    }
    _updateSimGauge(0);
    return;
  }

  console.log(`[Pairwise] Fetching comparison: ${label1} ↔ ${label2}`);

  try {
    const res = await fetch(
      `/speakers/centroid-comparison?label1=${encodeURIComponent(label1)}&label2=${encodeURIComponent(label2)}`,
    );
    const data = await res.json();

    if (data.error) {
      throw new Error(data.error);
    }

    pairwiseData = data;
    _renderPairwiseContent(label1, label2, data);
    _updateSimGauge(data.comparison?.cosine_similarity || 0);

    console.log("[Pairwise] Comparison rendered successfully");
  } catch (e) {
    console.error("[Pairwise] Error fetching comparison:", e);
    const content = document.getElementById("pairwiseContent");
    if (content) {
      content.innerHTML = `
        <div class="error-message">⚠️ Failed to load comparison: ${e.message}</div>`;
    }
  }
}

/**
 * Show empty state when no speakers selected
 */
function showEmptyPairwise() {
  const content = document.getElementById("pairwiseContent");
  if (content) {
    content.innerHTML = `
      <div class="empty-comparison">
        <div class="icon">🔍</div>
        <h3>Select Two Speakers</h3>
        <p>Choose two speakers above to see detailed pairwise comparison including segment-by-segment analysis, embedding visualization, and dimension differences.</p>
      </div>`;
  }
  _updateSimGauge(0);
}

/**
 * Switch between pairwise sub-tabs
 * @param {string} tabName - 'overview' | 'segments' | 'embeddings' | 'dimensions'
 */
function switchPairwiseTab(tabName) {
  // Update tab button states
  document
    .querySelectorAll("#pairwiseTabs .comparison-tab-btn")
    .forEach((btn) => {
      btn.classList.remove("active");
    });
  const activeBtn = document.querySelector(
    `#pairwiseTabs .comparison-tab-btn[onclick*="${tabName}"]`,
  );
  if (activeBtn) activeBtn.classList.add("active");

  // Show/hide tab contents
  document
    .querySelectorAll("#pairwiseContent .comparison-tab-content")
    .forEach((tab) => {
      tab.classList.remove("active");
    });
  const targetTab = document.getElementById(`pairwiseTab-${tabName}`);
  if (targetTab) targetTab.classList.add("active");

  // Resize charts if visible
  setTimeout(() => {
    Object.values(pairwiseCharts).forEach((chart) => {
      if (chart && chart.resize) chart.resize();
    });
  }, 100);

  console.log(`[Pairwise] Switched to tab: ${tabName}`);
}

/**
 * Destroy all pairwise charts
 */
function destroyPairwiseCharts() {
  Object.keys(pairwiseCharts).forEach((key) => {
    if (pairwiseCharts[key]) {
      pairwiseCharts[key].destroy();
      pairwiseCharts[key] = null;
    }
  });
  console.log("[Pairwise] All charts destroyed");
}

// ===== PRIVATE FUNCTIONS =====

/**
 * Update the circular similarity gauge
 * @private
 */
function _updateSimGauge(similarity) {
  const gauge = document.getElementById("similarityGauge");
  if (!gauge) return;

  const pct = Math.round(similarity * 100);
  let color;
  if (similarity >= 0.8)
    color = "#ef4444"; // Red - high similarity
  else if (similarity >= 0.5)
    color = "#eab308"; // Yellow - medium
  else color = "#22c55e"; // Green - low (well separated)

  gauge.style.setProperty("--similarity-pct", pct);
  gauge.style.setProperty("--similarity-color", color);
  const span = gauge.querySelector("span");
  if (span) span.textContent = pct + "%";
}

/**
 * Render the full pairwise comparison content with tabs
 * @private
 */
function _renderPairwiseContent(label1, label2, data) {
  destroyPairwiseCharts();

  const comparison = data.comparison || {};
  const similarity = comparison.cosine_similarity || 0;
  const wouldMerge = similarity >= (comparison.merge_threshold || 0.85);

  const color1 = typeof getColor === "function" ? getColor(0) : "#3b82f6";
  const color2 = typeof getColor === "function" ? getColor(1) : "#f97316";

  const content = document.getElementById("pairwiseContent");
  if (!content) return;

  content.innerHTML = `
    <!-- Sub-tab Navigation -->
    <div class="comparison-tabs" id="pairwiseTabs">
      <button class="comparison-tab-btn active" onclick="switchPairwiseTab('overview')">📊 Overview</button>
      <button class="comparison-tab-btn" onclick="switchPairwiseTab('segments')">📝 Segment View</button>
      <button class="comparison-tab-btn" onclick="switchPairwiseTab('embeddings')">🔬 Embedding Plot</button>
      <button class="comparison-tab-btn" onclick="switchPairwiseTab('dimensions')">📏 Dimension Diff</button>
    </div>
    
    <!-- Tab: Overview -->
    <div class="comparison-tab-content active" id="pairwiseTab-overview">
      <div class="plot-container">
        <h3>Comparison Overview: ${label1} ↔ ${label2}</h3>
        <div style="display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; margin-top:16px;">
          <div class="mini-card">
            <div class="mini-icon ${wouldMerge ? "bg-red" : "bg-green"}">${wouldMerge ? "⚠️" : "✅"}</div>
            <div class="mini-info">
              <div class="mini-value">${(similarity * 100).toFixed(1)}%</div>
              <div class="mini-label">Cosine Similarity</div>
            </div>
          </div>
          <div class="mini-card">
            <div class="mini-icon bg-blue">📏</div>
            <div class="mini-info">
              <div class="mini-value">${((1 - similarity) * 100).toFixed(1)}%</div>
              <div class="mini-label">Cosine Distance</div>
            </div>
          </div>
          <div class="mini-card">
            <div class="mini-icon bg-cyan">📐</div>
            <div class="mini-info">
              <div class="mini-value">${(comparison.euclidean_distance || 0).toFixed(4)}</div>
              <div class="mini-label">Euclidean Distance</div>
            </div>
          </div>
          <div class="mini-card">
            <div class="mini-icon ${wouldMerge ? "bg-red" : "bg-green"}">🔄</div>
            <div class="mini-info">
              <div class="mini-value">${wouldMerge ? "Would Merge" : "Separate"}</div>
              <div class="mini-label">Threshold: ${((comparison.merge_threshold || 0.85) * 100).toFixed(0)}%</div>
            </div>
          </div>
        </div>
        
        <!-- Speaker Detail Cards -->
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-top:20px;">
          <div class="speaker-panel speaker1-panel">
            <div class="speaker-panel-header">
              <span class="speaker-dot" style="background:${color1};"></span>
              <h4>${label1}</h4>
            </div>
            <div style="font-size:13px;color:var(--text-secondary);line-height:1.8;">
              <div>Segments: <strong>${comparison.speaker1_segments || "N/A"}</strong></div>
              <div>Centroid Quality: <strong>${((comparison.speaker1_quality || 0) * 100).toFixed(0)}%</strong></div>
              <div>Centroid Norm: <strong>${(comparison.speaker1_norm || 0).toFixed(4)}</strong></div>
            </div>
          </div>
          <div class="speaker-panel speaker2-panel">
            <div class="speaker-panel-header">
              <span class="speaker-dot" style="background:${color2};"></span>
              <h4>${label2}</h4>
            </div>
            <div style="font-size:13px;color:var(--text-secondary);line-height:1.8;">
              <div>Segments: <strong>${comparison.speaker2_segments || "N/A"}</strong></div>
              <div>Centroid Quality: <strong>${((comparison.speaker2_quality || 0) * 100).toFixed(0)}%</strong></div>
              <div>Centroid Norm: <strong>${(comparison.speaker2_norm || 0).toFixed(4)}</strong></div>
            </div>
          </div>
        </div>
      </div>
    </div>
    
    <!-- Tab: Segments -->
    <div class="comparison-tab-content" id="pairwiseTab-segments">
      <div class="segment-tabs-grid">
        <div class="speaker-panel speaker1-panel">
          <div class="speaker-panel-header">
            <span class="speaker-dot" style="background:${color1};"></span>
            <h4>${label1} - Segments</h4>
          </div>
          <div class="segment-list" id="segmentsList1">
            <div style="text-align:center;padding:20px;color:var(--text-secondary);">Loading segments...</div>
          </div>
        </div>
        <div class="speaker-panel speaker2-panel">
          <div class="speaker-panel-header">
            <span class="speaker-dot" style="background:${color2};"></span>
            <h4>${label2} - Segments</h4>
          </div>
          <div class="segment-list" id="segmentsList2">
            <div style="text-align:center;padding:20px;color:var(--text-secondary);">Loading segments...</div>
          </div>
        </div>
      </div>
    </div>
    
    <!-- Tab: Embeddings -->
    <div class="comparison-tab-content" id="pairwiseTab-embeddings">
      <div class="plot-container">
        <h3>Embedding Vector Comparison</h3>
        <div class="chart-wrapper large">
          <canvas id="embeddingComparisonChart"></canvas>
        </div>
      </div>
    </div>
    
    <!-- Tab: Dimensions -->
    <div class="comparison-tab-content" id="pairwiseTab-dimensions">
      <div class="plot-container">
        <h3>Top Dimension Differences</h3>
        <div class="chart-wrapper large">
          <canvas id="dimensionDiffChart"></canvas>
        </div>
        <div class="dimension-diff-list" id="dimensionDiffList" style="margin-top:16px;"></div>
      </div>
    </div>
  `;

  // Load segment data asynchronously
  _loadSegmentData(label1, label2, data);

  // Render charts with slight delay for DOM to settle
  setTimeout(
    () => _renderEmbeddingChart(label1, label2, data, color1, color2),
    150,
  );
  setTimeout(
    () => _renderDimensionDiffChart(label1, label2, data, color1, color2),
    150,
  );
}

/**
 * Load and render segment data for both speakers
 * @private
 */
function _loadSegmentData(label1, label2, data) {
  const list1 = document.getElementById("segmentsList1");
  const list2 = document.getElementById("segmentsList2");
  if (!list1 || !list2) return;

  const segments1 = data.segments?.[label1] || [];
  const segments2 = data.segments?.[label2] || [];

  if (segments1.length === 0 && segments2.length === 0) {
    list1.innerHTML =
      '<div style="text-align:center;padding:20px;color:var(--text-secondary);">No segment data available</div>';
    list2.innerHTML =
      '<div style="text-align:center;padding:20px;color:var(--text-secondary);">No segment data available</div>';
    return;
  }

  const renderSegment = (seg, i) => {
    const sim = seg.similarity_to_own_centroid || 0;
    const simClass = sim >= 0.7 ? "high" : sim >= 0.4 ? "medium" : "low";
    return `
      <div class="segment-item" title="Segment ${i + 1}: ${(seg.start || 0).toFixed(1)}s - ${(seg.end || 0).toFixed(1)}s">
        <span style="font-size:16px;">🎤</span>
        <span class="seg-time">${(seg.start || 0).toFixed(1)}s</span>
        <span style="flex:1;">Segment ${i + 1}</span>
        <span class="seg-similarity ${simClass}">${(sim * 100).toFixed(0)}%</span>
      </div>`;
  };

  list1.innerHTML =
    segments1.map(renderSegment).join("") ||
    '<div style="text-align:center;padding:20px;color:var(--text-secondary);">No segments</div>';
  list2.innerHTML =
    segments2.map(renderSegment).join("") ||
    '<div style="text-align:center;padding:20px;color:var(--text-secondary);">No segments</div>';
}

/**
 * Render the embedding comparison line chart
 * @private
 */
function _renderEmbeddingChart(label1, label2, data, color1, color2) {
  const canvas = document.getElementById("embeddingComparisonChart");
  if (!canvas) return;

  const ctx = canvas.getContext("2d");
  const comparison = data.comparison || {};
  const c1 = (comparison.speaker1_centroid_vector || []).slice(0, 50);
  const c2 = (comparison.speaker2_centroid_vector || []).slice(0, 50);

  if (c1.length === 0 && c2.length === 0) {
    canvas.parentElement.innerHTML =
      '<div class="empty-state"><p>No centroid data available</p></div>';
    return;
  }

  const labels = Array.from(
    { length: Math.max(c1.length, c2.length) },
    (_, i) => `D${i}`,
  );

  if (pairwiseCharts.embedding) pairwiseCharts.embedding.destroy();

  pairwiseCharts.embedding = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: label1,
          data: c1,
          borderColor: color1,
          backgroundColor: color1 + "22",
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.1,
          fill: false,
        },
        {
          label: label2,
          data: c2,
          borderColor: color2,
          backgroundColor: color2 + "22",
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.1,
          fill: false,
          borderDash: [5, 5],
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "bottom",
          labels: { color: "#94a3b8", usePointStyle: true, padding: 16 },
        },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: ${ctx.raw.toFixed(6)}`,
          },
        },
        title: {
          display: true,
          text: "Centroid Vector Overlay (First 50 Dimensions)",
          color: "#94a3b8",
          font: { size: 13 },
        },
      },
      scales: {
        x: {
          grid: { color: "rgba(148,163,184,0.06)" },
          ticks: {
            color: "#64748b",
            maxTicksLimit: 10,
            callback: (v, i) => (i % 5 === 0 ? labels[i] : ""),
          },
          title: { display: true, text: "Dimension", color: "#94a3b8" },
        },
        y: {
          grid: { color: "rgba(148,163,184,0.06)" },
          ticks: { color: "#64748b" },
          title: { display: true, text: "Value", color: "#94a3b8" },
        },
      },
    },
  });
}

/**
 * Render dimension difference bar chart and list
 * @private
 */
function _renderDimensionDiffChart(label1, label2, data, color1, color2) {
  const comparison = data.comparison || {};
  const topDims = comparison.top_different_dimensions || [];

  // --- Bar Chart ---
  const canvas = document.getElementById("dimensionDiffChart");
  if (canvas && topDims.length > 0) {
    const ctx = canvas.getContext("2d");
    if (pairwiseCharts.dimDiff) pairwiseCharts.dimDiff.destroy();

    pairwiseCharts.dimDiff = new Chart(ctx, {
      type: "bar",
      data: {
        labels: topDims.slice(0, 15).map((d) => `D${d.dimension}`),
        datasets: [
          {
            label: label1,
            data: topDims.slice(0, 15).map((d) => d.value_speaker1),
            backgroundColor: color1 + "CC",
            borderColor: color1,
            borderWidth: 1,
            borderRadius: 4,
          },
          {
            label: label2,
            data: topDims.slice(0, 15).map((d) => d.value_speaker2),
            backgroundColor: color2 + "CC",
            borderColor: color2,
            borderWidth: 1,
            borderRadius: 4,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: "bottom",
            labels: { color: "#94a3b8", usePointStyle: true, padding: 16 },
          },
          tooltip: {
            callbacks: {
              label: (ctx) => `${ctx.dataset.label}: ${ctx.raw.toFixed(6)}`,
            },
          },
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: { color: "#94a3b8", maxRotation: 45, font: { size: 10 } },
          },
          y: {
            grid: { color: "rgba(148,163,184,0.06)" },
            ticks: { color: "#64748b" },
          },
        },
      },
    });
  }

  // --- Difference List ---
  const listEl = document.getElementById("dimensionDiffList");
  if (!listEl) return;

  if (topDims.length === 0) {
    listEl.innerHTML =
      '<div style="text-align:center;padding:20px;color:var(--text-secondary);">No dimension data available</div>';
    return;
  }

  const maxAbs = Math.max(
    ...topDims.map((d) =>
      Math.max(Math.abs(d.value_speaker1), Math.abs(d.value_speaker2)),
    ),
    0.001,
  );

  listEl.innerHTML = topDims
    .slice(0, 15)
    .map((d) => {
      const pct1 = ((Math.abs(d.value_speaker1) / maxAbs) * 50).toFixed(1);
      const pct2 = ((Math.abs(d.value_speaker2) / maxAbs) * 50).toFixed(1);
      const diffSign = d.diff >= 0 ? "positive" : "negative";
      return `
      <div class="dimension-diff-item">
        <span class="dim-label">D${d.dimension}</span>
        <div class="dim-bar-track">
          <div class="dim-bar-fill speaker1-bar" style="width:${pct1}%;"></div>
          <div class="dim-bar-fill speaker2-bar" style="width:${pct2}%;"></div>
        </div>
        <span class="dim-diff-value ${diffSign}">
          Δ ${d.diff >= 0 ? "+" : ""}${d.diff.toFixed(4)}
        </span>
      </div>`;
    })
    .join("");
}
