// ============================================================
// dimension_diff_view.js
// Dimension Difference Analysis Module
// Dependencies: Chart.js 4.4.0+
// Requires global: allData, getColor()
// ============================================================

let dimDiffCharts = {};

/**
 * Initialize dimension difference view
 * @param {Array} speakerLabels - Available speaker labels
 */
function initDimensionDiffView(speakerLabels) {
  console.log(
    "[DimensionDiff] Initializing view with",
    speakerLabels?.length || 0,
    "speakers",
  );

  const speaker1Select = document.getElementById("dimDiffSpeaker1");
  const speaker2Select = document.getElementById("dimDiffSpeaker2");

  if (!speaker1Select || !speaker2Select) {
    console.warn("[DimensionDiff] Speaker selectors not found");
    return;
  }

  // Populate dropdowns
  [speaker1Select, speaker2Select].forEach((select) => {
    select.innerHTML = '<option value="">-- Select Speaker --</option>';
    if (speakerLabels) {
      speakerLabels.forEach((label) => {
        select.innerHTML += `<option value="${label}">${label}</option>`;
      });
    }
  });

  // Set initial values from URL params
  const params = getQueryParams();
  if (params.speaker1) speaker1Select.value = params.speaker1;
  if (params.speaker2) speaker2Select.value = params.speaker2;

  // Auto-load comparison if both speakers are set
  if (speaker1Select.value && speaker2Select.value) {
    loadDimensionDiff();
  }

  console.log("[DimensionDiff] View initialized");
}

/**
 * Load dimension difference data for selected speakers
 */
async function loadDimensionDiff() {
  const speaker1 = document.getElementById("dimDiffSpeaker1")?.value;
  const speaker2 = document.getElementById("dimDiffSpeaker2")?.value;

  if (!speaker1 || !speaker2) {
    showDimDiffEmpty("Select two speakers to compare dimensions");
    return;
  }

  if (speaker1 === speaker2) {
    showDimDiffEmpty("Select different speakers to compare");
    return;
  }

  console.log(`[DimensionDiff] Loading comparison: ${speaker1} vs ${speaker2}`);

  try {
    const response = await fetch(
      `/speakers/centroid-comparison?label1=${encodeURIComponent(speaker1)}&label2=${encodeURIComponent(speaker2)}`,
    );

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();

    if (data.error) {
      showDimDiffError(data.error);
      return;
    }

    renderDimensionDiffResults(data, speaker1, speaker2);
    console.log("[DimensionDiff] Comparison loaded successfully");
  } catch (error) {
    console.error("[DimensionDiff] Failed to load comparison:", error);
    showDimDiffError(`Failed to load comparison: ${error.message}`);
  }
}

/**
 * Render dimension difference results
 */
function renderDimensionDiffResults(comparisonData, speaker1, speaker2) {
  const container = document.getElementById("dimDiffResults");
  if (!container) {
    console.warn("[DimensionDiff] Results container not found");
    return;
  }

  const comparison = comparisonData.comparison;
  if (!comparison || !comparison.top_different_dimensions) {
    showDimDiffEmpty("No dimension data available for these speakers");
    return;
  }

  const topDims = comparison.top_different_dimensions;
  const maxDisplay = 20;
  const displayDims = topDims.slice(0, maxDisplay);

  // Build results HTML
  let html = buildDimDiffSummary(comparison, speaker1, speaker2);
  html += buildDimDiffChart(displayDims, speaker1, speaker2);
  html += buildDimDiffTable(displayDims, speaker1, speaker2, comparison);

  container.innerHTML = html;

  // Render the chart
  setTimeout(() => {
    renderDimDiffChart(displayDims, speaker1, speaker2);
  }, 100);
}

/**
 * Build summary section
 */
function buildDimDiffSummary(comparison, speaker1, speaker2) {
  const totalDimDiff = comparison.top_different_dimensions.reduce(
    (sum, d) => sum + Math.abs(d.diff),
    0,
  );
  const avgDiff =
    comparison.top_different_dimensions.length > 0
      ? totalDimDiff / comparison.top_different_dimensions.length
      : 0;

  return `
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-bottom: 20px;">
      <div class="mini-card">
        <div class="mini-icon bg-blue">📐</div>
        <div class="mini-info">
          <div class="mini-value">${comparison.cosine_similarity ? (comparison.cosine_similarity * 100).toFixed(1) + "%" : "N/A"}</div>
          <div class="mini-label">Cosine Similarity</div>
        </div>
      </div>
      <div class="mini-card">
        <div class="mini-icon bg-green">📏</div>
        <div class="mini-info">
          <div class="mini-value">${comparison.euclidean_distance ? comparison.euclidean_distance.toFixed(4) : "N/A"}</div>
          <div class="mini-label">Euclidean Distance</div>
        </div>
      </div>
      <div class="mini-card">
        <div class="mini-icon bg-yellow">🎯</div>
        <div class="mini-info">
          <div class="mini-value">${avgDiff.toFixed(4)}</div>
          <div class="mini-label">Avg Dimension Diff</div>
        </div>
      </div>
      <div class="mini-card">
        <div class="mini-icon ${comparison.would_merge ? "bg-red" : "bg-green"}">${comparison.would_merge ? "⚠️" : "✅"}</div>
        <div class="mini-info">
          <div class="mini-value">${comparison.would_merge ? "Would Merge" : "Separate"}</div>
          <div class="mini-label">Status (${((comparison.merge_threshold || 0.85) * 100).toFixed(0)}% threshold)</div>
        </div>
      </div>
    </div>`;
}

/**
 * Build chart container
 */
function buildDimDiffChart(displayDims, speaker1, speaker2) {
  return `
    <div class="plot-container">
      <h3>📊 Top Dimension Differences</h3>
      <div class="chart-wrapper" style="height: 400px;">
        <canvas id="dimDiffChart"></canvas>
      </div>
    </div>`;
}

/**
 * Build comparison table
 */
function buildDimDiffTable(displayDims, speaker1, speaker2, comparison) {
  const maxAbsVal = Math.max(
    ...displayDims.map((d) =>
      Math.max(Math.abs(d.value_speaker1), Math.abs(d.value_speaker2)),
    ),
    0.001,
  );

  let tableHtml = `
    <div class="plot-container" style="margin-top: 16px;">
      <h3>📋 Dimension Details (Top ${displayDims.length})</h3>
      <div style="overflow-x: auto; max-height: 500px; overflow-y: auto;">
        <table class="data-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Dimension</th>
              <th>${speaker1}</th>
              <th>${speaker2}</th>
              <th>Visual Comparison</th>
              <th>Difference</th>
            </tr>
          </thead>
          <tbody>`;

  displayDims.forEach((d, i) => {
    const absDiff = Math.abs(d.diff);
    const maxDiff = Math.max(...displayDims.map((x) => Math.abs(x.diff)));
    const pct1 = ((Math.abs(d.value_speaker1) / maxAbsVal) * 100).toFixed(0);
    const pct2 = ((Math.abs(d.value_speaker2) / maxAbsVal) * 100).toFixed(0);

    let diffClass = "";
    if (absDiff >= maxDiff * 0.7) diffClass = "bad";
    else if (absDiff >= maxDiff * 0.3) diffClass = "warning";
    else diffClass = "good";

    const rowBg = absDiff >= maxDiff * 0.7 ? "rgba(239, 68, 68, 0.05)" : "";

    tableHtml += `
      <tr style="background: ${rowBg};">
        <td><strong>${i + 1}</strong></td>
        <td><strong>D${d.dimension}</strong></td>
        <td style="font-family: monospace; color: var(--accent-blue);">${d.value_speaker1.toFixed(5)}</td>
        <td style="font-family: monospace; color: var(--accent-orange);">${d.value_speaker2.toFixed(5)}</td>
        <td>
          <div style="display: flex; align-items: center; gap: 4px;">
            <div style="flex: 1; height: 8px; background: var(--border-color); border-radius: 4px; overflow: hidden;">
              <div style="width: ${pct1}%; height: 100%; background: var(--accent-blue); float: left;"></div>
            </div>
            <div style="flex: 1; height: 8px; background: var(--border-color); border-radius: 4px; overflow: hidden;">
              <div style="width: ${pct2}%; height: 100%; background: var(--accent-orange); float: left;"></div>
            </div>
          </div>
        </td>
        <td>
          <span class="metric-badge ${diffClass}">
            ${d.diff >= 0 ? "+" : ""}${d.diff.toFixed(4)}
          </span>
        </td>
      </tr>`;
  });

  tableHtml += "</tbody></table></div></div>";
  return tableHtml;
}

/**
 * Render the dimension difference chart
 */
function renderDimDiffChart(displayDims, speaker1, speaker2) {
  const canvas = document.getElementById("dimDiffChart");
  if (!canvas) return;

  const ctx = canvas.getContext("2d");

  // Destroy existing chart
  if (dimDiffCharts.main) {
    dimDiffCharts.main.destroy();
  }

  const labels = displayDims.map((d) => `D${d.dimension}`);

  dimDiffCharts.main = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [
        {
          label: speaker1,
          data: displayDims.map((d) => d.value_speaker1),
          backgroundColor: "rgba(59, 130, 246, 0.7)",
          borderColor: "#3b82f6",
          borderWidth: 1,
          borderRadius: 4,
          order: 2,
        },
        {
          label: speaker2,
          data: displayDims.map((d) => d.value_speaker2),
          backgroundColor: "rgba(249, 115, 22, 0.7)",
          borderColor: "#f97316",
          borderWidth: 1,
          borderRadius: 4,
          order: 2,
        },
        {
          label: "Absolute Difference",
          data: displayDims.map((d) => Math.abs(d.diff)),
          type: "line",
          borderColor: "#ef4444",
          backgroundColor: "transparent",
          borderWidth: 2,
          pointRadius: 3,
          pointBackgroundColor: "#ef4444",
          yAxisID: "y1",
          order: 1,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: "index",
        intersect: false,
      },
      plugins: {
        legend: {
          position: "bottom",
          labels: {
            color: "#94a3b8",
            usePointStyle: true,
            padding: 16,
          },
        },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              if (ctx.dataset.label === "Absolute Difference") {
                return `Δ: ${ctx.raw.toFixed(5)}`;
              }
              return `${ctx.dataset.label}: ${ctx.raw.toFixed(5)}`;
            },
          },
        },
        title: {
          display: true,
          text: `${speaker1} vs ${speaker2} - Top Dimension Differences`,
          color: "#94a3b8",
          font: { size: 14 },
        },
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: {
            color: "#94a3b8",
            maxRotation: 45,
            font: { size: 10 },
          },
        },
        y: {
          position: "left",
          grid: { color: "rgba(148,163,184,0.08)" },
          ticks: { color: "#64748b" },
          title: {
            display: true,
            text: "Value",
            color: "#94a3b8",
          },
        },
        y1: {
          position: "right",
          grid: { display: false },
          ticks: {
            color: "#ef4444",
            callback: (v) => v.toFixed(3),
          },
          title: {
            display: true,
            text: "|Difference|",
            color: "#ef4444",
          },
        },
      },
    },
  });

  console.log(
    `[DimensionDiff] Chart rendered with ${displayDims.length} dimensions`,
  );
}

/**
 * Show empty state
 */
function showDimDiffEmpty(message) {
  const container = document.getElementById("dimDiffResults");
  if (!container) return;

  container.innerHTML = `
    <div class="empty-state">
      <div class="icon">📏</div>
      <p>${message || "Select two speakers to see dimension differences"}</p>
    </div>`;
}

/**
 * Show error state
 */
function showDimDiffError(error) {
  const container = document.getElementById("dimDiffResults");
  if (!container) return;

  container.innerHTML = `
    <div class="error-message">
      <p><strong>⚠️ Error</strong></p>
      <p>${error}</p>
    </div>`;
}

/**
 * Export dimension difference data as CSV
 */
function exportDimDiffCSV() {
  const speaker1 = document.getElementById("dimDiffSpeaker1")?.value;
  const speaker2 = document.getElementById("dimDiffSpeaker2")?.value;

  if (!speaker1 || !speaker2) {
    alert("Select two speakers first");
    return;
  }

  // Get data from the chart or DOM
  const rows = document.querySelectorAll(
    "#dimDiffResults .data-table tbody tr",
  );
  if (rows.length === 0) {
    alert("No data to export");
    return;
  }

  let csv = `Rank,Dimension,${speaker1},${speaker2},Difference\n`;
  rows.forEach((row, i) => {
    const cells = row.querySelectorAll("td");
    const dim = cells[1]?.textContent.trim() || "";
    const val1 = cells[2]?.textContent.trim() || "";
    const val2 = cells[3]?.textContent.trim() || "";
    const diff = cells[5]?.textContent.trim() || "";
    csv += `${i + 1},${dim},${val1},${val2},${diff}\n`;
  });

  const blob = new Blob([csv], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `dimension_diff_${speaker1}_vs_${speaker2}_${new Date().toISOString().slice(0, 10)}.csv`;
  link.click();
  URL.revokeObjectURL(url);

  console.log(`[DimensionDiff] Exported CSV for ${speaker1} vs ${speaker2}`);
}

/**
 * Destroy all dimension diff charts
 */
function destroyDimDiffCharts() {
  Object.keys(dimDiffCharts).forEach((key) => {
    if (dimDiffCharts[key]) {
      dimDiffCharts[key].destroy();
      dimDiffCharts[key] = null;
    }
  });
  console.log("[DimensionDiff] All charts destroyed");
}

// Expose globally
window.initDimensionDiffView = initDimensionDiffView;
window.loadDimensionDiff = loadDimensionDiff;
window.exportDimDiffCSV = exportDimDiffCSV;
window.destroyDimDiffCharts = destroyDimDiffCharts;
