// ============================================================
// health_diagnostics.js
// Speaker System Health & Diagnostics Dashboard
// Dependencies: Chart.js 4.4.0+
// Requires global: allData
// ============================================================

let healthCharts = {};

/**
 * Render the complete health dashboard
 * @param {object} data - The full allData object
 */
function renderHealthDashboard(data) {
  const { status } = data;
  if (!status) {
    console.warn("[Health] No status data available");
    return;
  }

  _renderHealthContent(status);
  _renderHealthCategoriesChart(status);
  _renderMergeHistory(status);
  _renderRejectionChart(status);

  console.log("[Health] Dashboard rendered successfully");
}

/**
 * Destroy all health charts
 */
function destroyHealthCharts() {
  Object.keys(healthCharts).forEach((key) => {
    if (healthCharts[key]) {
      healthCharts[key].destroy();
      healthCharts[key] = null;
    }
  });
  console.log("[Health] All charts destroyed");
}

// ===== PRIVATE FUNCTIONS =====

/**
 * Render the main health status content
 * @private
 */
function _renderHealthContent(status) {
  const container = document.getElementById("healthContent");
  if (!container) return;

  const alerts = status.alerts || [];
  const totalSpeakers = status.total_speakers || 0;
  const matureSpeakers = status.mature_speakers || 0;
  const youngSpeakers = status.young_speakers || 0;
  const orphanSpeakers = status.orphan_speakers || 0;
  const rejectionRate = status.centroids?.rejection_rate || 0;
  const totalSegments = status.centroids?.total_segments_processed || 0;
  const totalRejected = status.centroids?.total_updates_rejected || 0;
  const missingIds = status.missing_speaker_ids || [];

  // Determine overall health
  let healthStatus, healthIcon;
  if (alerts[0]?.includes("✅")) {
    healthStatus = "good";
    healthIcon = "✅";
  } else if (alerts.length <= 2) {
    healthStatus = "warning";
    healthIcon = "⚠️";
  } else {
    healthStatus = "bad";
    healthIcon = "🚨";
  }

  container.innerHTML = `
    <!-- Metric Cards -->
    <div style="display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px;">
      <div class="mini-card">
        <div class="mini-icon bg-${healthStatus === "good" ? "green" : healthStatus === "warning" ? "yellow" : "red"}">
          ${healthIcon}
        </div>
        <div class="mini-info">
          <div class="mini-value">${alerts[0] || "Unknown"}</div>
          <div class="mini-label">System Status</div>
        </div>
      </div>
      <div class="mini-card">
        <div class="mini-icon bg-blue">👥</div>
        <div class="mini-info">
          <div class="mini-value">${totalSpeakers}</div>
          <div class="mini-label">Total Speakers</div>
        </div>
      </div>
      <div class="mini-card">
        <div class="mini-icon bg-green">✅</div>
        <div class="mini-info">
          <div class="mini-value">${matureSpeakers}</div>
          <div class="mini-label">Mature (5+ segs)</div>
        </div>
      </div>
      <div class="mini-card">
        <div class="mini-icon bg-yellow">🌱</div>
        <div class="mini-info">
          <div class="mini-value">${youngSpeakers}</div>
          <div class="mini-label">Young (1-2 segs)</div>
        </div>
      </div>
      <div class="mini-card">
        <div class="mini-icon bg-red">👻</div>
        <div class="mini-info">
          <div class="mini-value">${orphanSpeakers}</div>
          <div class="mini-label">Orphan (inactive)</div>
        </div>
      </div>
      <div class="mini-card">
        <div class="mini-icon bg-purple">🛡️</div>
        <div class="mini-info">
          <div class="mini-value">${(rejectionRate * 100).toFixed(1)}%</div>
          <div class="mini-label">Rejection Rate</div>
        </div>
      </div>
    </div>
    
    <!-- Alerts List -->
    <div style="margin-top:16px; padding:12px; background:rgba(0,0,0,0.2); border-radius:8px;">
      <strong style="font-size:13px;">📋 System Alerts:</strong>
      <ul style="margin:8px 0 0 16px; font-size:13px; color:var(--text-secondary); line-height:1.8;">
        ${alerts.map((a) => `<li>${a}</li>`).join("")}
      </ul>
    </div>
    
    <!-- Quick Stats -->
    <div style="margin-top:12px; display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:8px; font-size:12px; color:var(--text-secondary);">
      <div>📝 Total Segments: <strong style="color:var(--text-primary);">${totalSegments}</strong></div>
      <div>🚫 Rejected Updates: <strong style="color:var(--text-primary);">${totalRejected}</strong></div>
      <div>🆔 Missing IDs: <strong style="color:var(--text-primary);">${missingIds.join(", ") || "None"}</strong></div>
      <div>📊 Young/Mature Ratio: <strong style="color:var(--text-primary);">${status.young_to_mature_ratio || 0}</strong></div>
    </div>
  `;
}

/**
 * Render speaker categories doughnut chart
 * @private
 */
function _renderHealthCategoriesChart(status) {
  const canvas = document.getElementById("healthCategoriesChart");
  if (!canvas) return;

  const ctx = canvas.getContext("2d");
  const categories = status.categories || {};

  const mature = (categories.mature || []).length;
  const young = (categories.young || []).length;
  const activeYoung = (categories.active_young || []).length;
  const orphan = (categories.orphan || []).length;
  const newborn = (categories.newborn || []).length;
  const total = status.total_speakers || 1;

  if (healthCharts.categories) healthCharts.categories.destroy();

  healthCharts.categories = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: [
        "Mature (5+)",
        "Growing (3-4)",
        "Young (1-2)",
        "Orphan",
        "Newborn",
      ],
      datasets: [
        {
          data: [mature, activeYoung, young, orphan, newborn],
          backgroundColor: [
            "#22c55e",
            "#3b82f6",
            "#eab308",
            "#ef4444",
            "#a855f7",
          ],
          borderColor: "#1e293b",
          borderWidth: 3,
          hoverBorderColor: "#334155",
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "bottom",
          labels: {
            color: "#94a3b8",
            padding: 10,
            usePointStyle: true,
            font: { size: 11 },
          },
        },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const pct = ((ctx.raw / total) * 100).toFixed(0);
              return `${ctx.label}: ${ctx.raw} speaker(s) (${pct}%)`;
            },
          },
        },
      },
    },
  });
}

/**
 * Render merge history table
 * @private
 */
function _renderMergeHistory(status) {
  const container = document.getElementById("mergeHistoryContent");
  if (!container) return;

  const mergeHistory = status.merge_history || [];

  if (mergeHistory.length === 0) {
    container.innerHTML = `
      <div class="empty-state" style="padding:40px;">
        <div class="icon">✅</div>
        <p>No merges performed yet</p>
        <span style="font-size:12px;color:var(--text-secondary);">Speakers are well separated</span>
      </div>`;
    return;
  }

  container.innerHTML = `
    <table class="data-table">
      <thead>
        <tr>
          <th>Type</th>
          <th>Source</th>
          <th>→ Target</th>
          <th>Similarity</th>
          <th>Time</th>
        </tr>
      </thead>
      <tbody>
        ${mergeHistory
          .slice(-20)
          .reverse()
          .map(
            (m) => `
          <tr>
            <td><span class="metric-badge info">${m.type || "merge"}</span></td>
            <td style="color:#ef4444;">${m.source || "?"}</td>
            <td style="color:#22c55e;"><strong>${m.target || "?"}</strong></td>
            <td>${m.similarity ? (m.similarity * 100).toFixed(1) + "%" : "N/A"}</td>
            <td style="font-size:11px;color:var(--text-secondary);">
              ${m.timestamp ? new Date(m.timestamp * 1000).toLocaleTimeString() : "N/A"}
            </td>
          </tr>
        `,
          )
          .join("")}
      </tbody>
    </table>
    ${mergeHistory.length > 20 ? `<div style="text-align:center;padding:8px;font-size:11px;color:var(--text-secondary);">Showing last 20 of ${mergeHistory.length} merges</div>` : ""}
  `;
}

/**
 * Render centroid update rejection analysis chart
 * @private
 */
function _renderRejectionChart(status) {
  const canvas = document.getElementById("rejectionChart");
  if (!canvas) return;

  const ctx = canvas.getContext("2d");
  const centroids = status.centroids || {};

  const totalProcessed = centroids.total_segments_processed || 0;
  const totalRejected = centroids.total_updates_rejected || 0;
  const totalAccepted = totalProcessed - totalRejected;
  const rejectionRate = centroids.rejection_rate || 0;

  if (healthCharts.rejection) healthCharts.rejection.destroy();

  healthCharts.rejection = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["Centroid Updates"],
      datasets: [
        {
          label: "Accepted ✓",
          data: [totalAccepted],
          backgroundColor: "#22c55eCC",
          borderColor: "#22c55e",
          borderWidth: 1,
          borderRadius: 6,
          borderSkipped: false,
        },
        {
          label: "Rejected ✗",
          data: [totalRejected],
          backgroundColor: "#ef4444CC",
          borderColor: "#ef4444",
          borderWidth: 1,
          borderRadius: 6,
          borderSkipped: false,
        },
      ],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "bottom",
          labels: { color: "#94a3b8", usePointStyle: true, padding: 12 },
        },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const pct = (
                (ctx.raw / Math.max(totalProcessed, 1)) *
                100
              ).toFixed(1);
              return `${ctx.dataset.label}: ${ctx.raw} (${pct}%)`;
            },
          },
        },
        title: {
          display: true,
          text: `Contamination Prevention — ${(rejectionRate * 100).toFixed(1)}% Rejection Rate`,
          color: "#94a3b8",
          font: { size: 13 },
        },
      },
      scales: {
        x: {
          stacked: true,
          grid: { color: "rgba(148,163,184,0.06)" },
          ticks: { color: "#64748b", stepSize: 1 },
          title: { display: true, text: "Number of Updates", color: "#94a3b8" },
        },
        y: {
          stacked: true,
          grid: { display: false },
          ticks: { color: "#94a3b8" },
        },
      },
    },
  });
}
