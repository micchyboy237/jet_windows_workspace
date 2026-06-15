// ============================================================
// independent_analysis.js
// Independent Speaker Analysis - Side-by-side embedding visualization
// Dependencies: Chart.js 4.4.0+
// Requires global: allData
// ============================================================

let indieCharts = {};

/**
 * Initialize independent analysis speaker selection dropdowns
 * @param {string[]} speakerLabels - Array of speaker label strings
 */
function initIndependentSelects(speakerLabels) {
  const select1 = document.getElementById("indieSpeaker1");
  const select2 = document.getElementById("indieSpeaker2");

  if (!select1 || !select2) {
    console.log("[IndieAnalysis] Select elements not found in DOM");
    return;
  }

  const current1 = select1.value;
  const current2 = select2.value;

  [select1, select2].forEach((select) => {
    select.innerHTML = '<option value="">-- Select Speaker --</option>';
    speakerLabels.forEach((label, i) => {
      const option = document.createElement("option");
      option.value = label;
      option.textContent = `${label} (${i + 1})`;
      select.appendChild(option);
    });
  });

  if (speakerLabels.includes(current1)) select1.value = current1;
  if (speakerLabels.includes(current2)) select2.value = current2;

  if (!select1.value && speakerLabels.length >= 1)
    select1.value = speakerLabels[0];
  if (!select2.value && speakerLabels.length >= 2)
    select2.value = speakerLabels[1];

  if (select1.value && select2.value) {
    updateIndependentAnalysis();
  }

  console.log(
    `[IndieAnalysis] Initialized with ${speakerLabels.length} speakers`,
  );
}

/**
 * Destroy all independent analysis charts
 */
function destroyIndieCharts() {
  Object.keys(indieCharts).forEach((key) => {
    if (indieCharts[key]) {
      indieCharts[key].destroy();
      indieCharts[key] = null;
    }
  });
  console.log("[IndieAnalysis] All charts destroyed");
}

/**
 * Update the independent analysis view based on current selections
 */
async function updateIndependentAnalysis() {
  const label1 = document.getElementById("indieSpeaker1")?.value;
  const label2 = document.getElementById("indieSpeaker2")?.value;
  if (!label1 && !label2) return;

  destroyIndieCharts();

  const color1 = "#3b82f6";
  const color2 = "#f97316";

  // Render speaker 1 embedding if selected
  if (label1 && allData) {
    const sp1 = allData.centroids?.centroids?.[label1];
    if (sp1) {
      const container1 = document.getElementById(
        "embedding-speaker1-container",
      );
      if (
        container1 &&
        container1._useComponent &&
        typeof renderSpeakerEmbedding === "function"
      ) {
        // Use the component's renderer
        console.log(
          "[IndieAnalysis] Using speaker_embedding_plot component for Speaker 1",
        );
        renderSpeakerEmbedding(
          "embedding-speaker1-container",
          {
            ...sp1,
            label: label1,
          },
          {
            color: color1,
            chartType: "line",
            showStats: true,
          },
        );
      } else if (typeof _renderSingleEmbedding === "function") {
        // Fallback to independent_analysis.js renderer
        _renderSingleEmbedding(
          "embedding-speaker1-container",
          label1,
          sp1,
          color1,
        );
      }
    }
  }

  // Render speaker 2 embedding if selected
  if (label2 && allData) {
    const sp2 = allData.centroids?.centroids?.[label2];
    if (sp2) {
      const container2 = document.getElementById(
        "embedding-speaker2-container",
      );
      if (
        container2 &&
        container2._useComponent &&
        typeof renderSpeakerEmbedding === "function"
      ) {
        console.log(
          "[IndieAnalysis] Using speaker_embedding_plot component for Speaker 2",
        );
        renderSpeakerEmbedding(
          "embedding-speaker2-container",
          {
            ...sp2,
            label: label2,
          },
          {
            color: color2,
            chartType: "line",
            showStats: true,
          },
        );
      } else if (typeof _renderSingleEmbedding === "function") {
        _renderSingleEmbedding(
          "embedding-speaker2-container",
          label2,
          sp2,
          color2,
        );
      }
    }
  }

  // If both selected and different, fetch comparison data
  if (label1 && label2 && label1 !== label2) {
    try {
      const res = await fetch(
        `/speakers/centroid-comparison?label1=${encodeURIComponent(label1)}&label2=${encodeURIComponent(label2)}`,
      );
      const data = await res.json();
      if (!data.error) {
        const similarity = data.comparison?.cosine_similarity || 0;

        // Similarity Gauge - use component if available
        const gaugeContainer = document.getElementById("indie-gauge-container");
        if (
          gaugeContainer &&
          gaugeContainer._useComponent &&
          typeof updateSimilarityGauge === "function"
        ) {
          console.log("[IndieAnalysis] Using similarity_gauge component");
          // The include already created the gauge structure, just update it
          const gaugeEl = gaugeContainer.querySelector(".sim-gauge-container");
          if (gaugeEl) {
            updateSimilarityGauge(
              gaugeEl.id || "indie-gauge-container",
              similarity,
              {
                speaker1: label1,
                speaker2: label2,
              },
            );
          } else {
            // Create new gauge
            const gauge = createSimilarityGauge("indie-gauge-main", {
              size: "lg",
            });
            gaugeContainer.innerHTML = "";
            gaugeContainer.appendChild(gauge);
            updateSimilarityGauge("indie-gauge-main", similarity, {
              speaker1: label1,
              speaker2: label2,
            });
          }
        } else if (typeof _renderIndieSimilarityGauge === "function") {
          _renderIndieSimilarityGauge(similarity, label1, label2);
        }

        // Dimension Diff - use component if available
        const dimDiffContainer = document.getElementById(
          "indie-dimdiff-content",
        );
        if (
          dimDiffContainer &&
          dimDiffContainer._useComponent &&
          typeof renderDimensionDiff === "function"
        ) {
          console.log("[IndieAnalysis] Using dimension_diff_view component");
          renderDimensionDiff("indie-dimdiff-content", data, {
            speaker1Label: label1,
            speaker2Label: label2,
            colors: { speaker1: color1, speaker2: color2 },
          });
        } else if (typeof _renderIndieDimensionDiff === "function") {
          _renderIndieDimensionDiff(data, label1, label2, color1, color2);
        }

        // Comparison table (always use the JS version since no component exists for this)
        if (typeof _renderIndieComparisonTable === "function") {
          _renderIndieComparisonTable(data, label1, label2, color1, color2);
        }
      }
    } catch (e) {
      console.error("[IndieAnalysis] Error fetching comparison:", e);
    }
  }
  console.log(`[IndieAnalysis] Updated: ${label1 || "?"} vs ${label2 || "?"}`);
}

// ===== PRIVATE FUNCTIONS =====

/**
 * Render a single speaker embedding chart in a container
 * @private
 */
function _renderSingleEmbedding(containerId, label, speakerData, color) {
  const container = document.getElementById(containerId);
  if (!container) return;

  const vector = speakerData.centroid_vector || [];
  const displayVector = vector.slice(0, 50);
  const chartId = `${containerId}-chart`;

  container.innerHTML = `
    <div class="plot-container">
      <div class="speaker-panel-header" style="margin-bottom:12px;">
        <span class="speaker-dot" style="background:${color};"></span>
        <h4>${label} - Embedding Vector</h4>
      </div>
      <div class="chart-wrapper" style="height:280px;">
        <canvas id="${chartId}"></canvas>
      </div>
      <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:8px; margin-top:12px; padding-top:12px; border-top:1px solid var(--border-color);">
        <div style="text-align:center;">
          <div style="font-size:16px;font-weight:700;color:var(--text-primary);">${speakerData.segment_count || 0}</div>
          <div style="font-size:10px;color:var(--text-secondary);">Segments</div>
        </div>
        <div style="text-align:center;">
          <div style="font-size:16px;font-weight:700;color:var(--text-primary);">${((speakerData.centroid_quality || 0) * 100).toFixed(0)}%</div>
          <div style="font-size:10px;color:var(--text-secondary);">Quality</div>
        </div>
        <div style="text-align:center;">
          <div style="font-size:16px;font-weight:700;color:var(--text-primary);">${(speakerData.centroid_norm || 0).toFixed(3)}</div>
          <div style="font-size:10px;color:var(--text-secondary);">Norm</div>
        </div>
        <div style="text-align:center;">
          <div style="font-size:16px;font-weight:700;color:var(--text-primary);">${(speakerData.active_duration || 0).toFixed(1)}s</div>
          <div style="font-size:10px;color:var(--text-secondary);">Active</div>
        </div>
      </div>
    </div>`;

  if (displayVector.length === 0) return;

  setTimeout(() => {
    const canvas = document.getElementById(chartId);
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const labels = displayVector.map((_, i) => `D${i}`);

    indieCharts[chartId] = new Chart(ctx, {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label,
            data: displayVector,
            borderColor: color,
            backgroundColor: color + "22",
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.1,
            fill: true,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: (ctx) => `${ctx.raw.toFixed(6)}` } },
        },
        scales: {
          x: {
            grid: { color: "rgba(148,163,184,0.06)" },
            ticks: {
              color: "#64748b",
              maxTicksLimit: 8,
              callback: (v, i) => (i % 5 === 0 ? labels[i] : ""),
            },
          },
          y: {
            grid: { color: "rgba(148,163,184,0.06)" },
            ticks: { color: "#64748b" },
          },
        },
      },
    });
  }, 100);
}

/**
 * Render similarity gauge between two speakers
 * @private
 */
function _renderIndieSimilarityGauge(similarity, label1, label2) {
  const container = document.getElementById("indie-gauge-container");
  if (!container) return;

  const pct = Math.round(similarity * 100);
  let color, badge, badgeClass;

  if (similarity >= 0.8) {
    color = "#ef4444";
    badge = "⚠️ Would Merge";
    badgeClass = "bad";
  } else if (similarity >= 0.5) {
    color = "#eab308";
    badge = "🔶 Borderline";
    badgeClass = "warning";
  } else {
    color = "#22c55e";
    badge = "✅ Well Separated";
    badgeClass = "good";
  }

  container.innerHTML = `
    <div style="text-align:center;">
      <div style="
        width:100px; height:100px; border-radius:50%;
        display:inline-flex; align-items:center; justify-content:center;
        font-size:22px; font-weight:700; position:relative;
        background:conic-gradient(${color} ${pct}%, var(--border-color) ${pct}%);
        margin-bottom:12px;
      ">
        <span style="
          position:relative; z-index:1; color:var(--text-primary);
          display:flex; align-items:center; justify-content:center;
          width:84px; height:84px; border-radius:50%; background:var(--bg-card);
        ">${pct}%</span>
      </div>
      <div style="font-size:14px;color:var(--text-secondary);">
        <strong style="color:${color};">${label1}</strong> 
        ↔ 
        <strong style="color:${color};">${label2}</strong>
      </div>
      <span class="metric-badge ${badgeClass}" style="margin-top:8px;display:inline-block;">
        ${badge}
      </span>
    </div>`;
}

/**
 * Render dimension difference chart and list
 * @private
 */
function _renderIndieDimensionDiff(data, label1, label2, color1, color2) {
  const container = document.getElementById("indie-dimdiff-content");
  if (!container) return;

  const topDims = data.comparison?.top_different_dimensions || [];

  if (topDims.length === 0) {
    container.innerHTML =
      '<div class="empty-state"><p>No dimension data available</p></div>';
    return;
  }

  const maxAbs = Math.max(
    ...topDims.map((d) =>
      Math.max(Math.abs(d.value_speaker1), Math.abs(d.value_speaker2)),
    ),
    0.001,
  );

  container.innerHTML = `
    <div class="chart-wrapper" style="height:300px;">
      <canvas id="indie-dimdiff-chart"></canvas>
    </div>
    <div style="max-height:300px; overflow-y:auto; margin-top:12px;">
      ${topDims
        .slice(0, 15)
        .map((d) => {
          const pct1 = ((Math.abs(d.value_speaker1) / maxAbs) * 50).toFixed(1);
          const pct2 = ((Math.abs(d.value_speaker2) / maxAbs) * 50).toFixed(1);
          return `
          <div class="dimension-diff-item">
            <span class="dim-label">D${d.dimension}</span>
            <div class="dim-bar-track">
              <div class="dim-bar-fill speaker1-bar" style="width:${pct1}%;"></div>
              <div class="dim-bar-fill speaker2-bar" style="width:${pct2}%;"></div>
            </div>
            <span class="dim-diff-value ${d.diff >= 0 ? "positive" : "negative"}">
              Δ ${d.diff >= 0 ? "+" : ""}${d.diff.toFixed(4)}
            </span>
          </div>`;
        })
        .join("")}
    </div>`;

  setTimeout(() => {
    const canvas = document.getElementById("indie-dimdiff-chart");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    indieCharts.indieDimDiff = new Chart(ctx, {
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
            borderRadius: 3,
          },
          {
            label: label2,
            data: topDims.slice(0, 15).map((d) => d.value_speaker2),
            backgroundColor: color2 + "CC",
            borderColor: color2,
            borderWidth: 1,
            borderRadius: 3,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: "bottom",
            labels: { color: "#94a3b8", usePointStyle: true, padding: 12 },
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
  }, 100);
}

/**
 * Render comparison summary table
 * @private
 */
function _renderIndieComparisonTable(data, label1, label2, color1, color2) {
  const container = document.getElementById("indie-comparison-table-content");
  if (!container) return;

  const c = data.comparison || {};
  const sim = c.cosine_similarity || 0;
  const wouldMerge = sim >= (c.merge_threshold || 0.85);

  container.innerHTML = `
    <table class="data-table">
      <thead>
        <tr>
          <th>Metric</th>
          <th>
            <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${color1};margin-right:6px;"></span>
            ${label1}
          </th>
          <th>
            <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${color2};margin-right:6px;"></span>
            ${label2}
          </th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><strong>Cosine Similarity</strong></td>
          <td colspan="2" style="text-align:center;">${(sim * 100).toFixed(2)}%</td>
          <td><span class="metric-badge ${wouldMerge ? "bad" : "good"}">${wouldMerge ? "⚠️ Merge Candidate" : "✅ Separate"}</span></td>
        </tr>
        <tr>
          <td>Cosine Distance</td>
          <td colspan="2" style="text-align:center;">${((1 - sim) * 100).toFixed(2)}%</td>
          <td>-</td>
        </tr>
        <tr>
          <td>Euclidean Distance</td>
          <td colspan="2" style="text-align:center;">${(c.euclidean_distance || 0).toFixed(4)}</td>
          <td>-</td>
        </tr>
        <tr>
          <td>Segments</td>
          <td>${c.speaker1_segments || "N/A"}</td>
          <td>${c.speaker2_segments || "N/A"}</td>
          <td>-</td>
        </tr>
        <tr>
          <td>Centroid Quality</td>
          <td>${((c.speaker1_quality || 0) * 100).toFixed(0)}%</td>
          <td>${((c.speaker2_quality || 0) * 100).toFixed(0)}%</td>
          <td>-</td>
        </tr>
        <tr>
          <td>Centroid Norm</td>
          <td>${(c.speaker1_norm || 0).toFixed(4)}</td>
          <td>${(c.speaker2_norm || 0).toFixed(4)}</td>
          <td>-</td>
        </tr>
      </tbody>
    </table>`;
}
