// ============================================================
// similarity_network.js
// Speaker Similarity Network & Distribution
// Dependencies: Chart.js 4.4.0+
// Requires global: allData, getColor()
// ============================================================

let networkCharts = {};

/**
 * Render the force-directed similarity network
 * @param {object} data - The full allData object
 */
function renderSimilarityNetwork(data) {
  const { centroids, distances } = data;
  const speakers = centroids?.centroids || {};
  const labels = Object.keys(speakers);
  const simMatrix = distances?.similarities || [];

  const wrapper = document.getElementById("networkWrapper");
  if (!wrapper) return;

  if (labels.length < 2) {
    wrapper.innerHTML =
      '<div class="empty-state"><div class="icon">🕸️</div><p>Need 2+ speakers for network visualization</p></div>';
    _renderEmptySimDist();
    return;
  }

  // Build edge connections
  const edges = [];
  const nodeWeights = labels.map((l) => speakers[l]?.segment_count || 1);
  const maxWeight = Math.max(...nodeWeights, 1);

  for (let i = 0; i < labels.length; i++) {
    for (let j = i + 1; j < labels.length; j++) {
      const sim = simMatrix[i]?.[j] ?? 0;
      if (sim > 0.1) {
        edges.push({
          from: labels[i],
          to: labels[j],
          similarity: sim,
          wouldMerge: sim >= 0.85,
        });
      }
    }
  }

  // Setup canvas
  wrapper.innerHTML =
    '<canvas id="networkChart" style="width:100%; height:100%;"></canvas>';
  const canvas = document.getElementById("networkChart");
  if (!canvas) return;

  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.parentElement.getBoundingClientRect();

  canvas.width = rect.width * dpr;
  canvas.height = 520 * dpr;
  canvas.style.width = rect.width + "px";
  canvas.style.height = "520px";
  ctx.scale(dpr, dpr);

  const W = rect.width;
  const H = 520;
  const cx = W / 2;
  const cy = H / 2;
  const radius = Math.min(W, H) * 0.35;

  // Position nodes in a circle
  const nodePositions = {};
  labels.forEach((label, i) => {
    const angle = (i / labels.length) * 2 * Math.PI - Math.PI / 2;
    nodePositions[label] = {
      x: cx + Math.cos(angle) * radius,
      y: cy + Math.sin(angle) * radius,
      weight: nodeWeights[i],
      color: typeof getColor === "function" ? getColor(i) : "#3b82f6",
      label,
    };
  });

  // Draw edges
  edges.forEach((edge) => {
    const from = nodePositions[edge.from];
    const to = nodePositions[edge.to];
    if (!from || !to) return;

    ctx.beginPath();
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(to.x, to.y);

    if (edge.wouldMerge) {
      ctx.strokeStyle = "rgba(239, 68, 68, 0.5)";
      ctx.lineWidth = 3;
      ctx.setLineDash([]);
    } else {
      const alpha = 0.1 + edge.similarity * 0.3;
      ctx.strokeStyle = `rgba(148, 163, 184, ${alpha.toFixed(2)})`;
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
    }
    ctx.stroke();
  });
  ctx.setLineDash([]);

  // Draw nodes
  labels.forEach((label, i) => {
    const pos = nodePositions[label];
    if (!pos) return;

    const nodeRadius = 12 + (pos.weight / maxWeight) * 20;

    // Glow effect for high-weight speakers
    if (pos.weight >= 5) {
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, nodeRadius + 4, 0, Math.PI * 2);
      ctx.fillStyle = pos.color + "22";
      ctx.fill();
    }

    // Main node circle
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, nodeRadius, 0, Math.PI * 2);
    ctx.fillStyle = pos.color;
    ctx.fill();
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 2;
    ctx.stroke();

    // Speaker label
    ctx.fillStyle = "#f1f5f9";
    ctx.font = "bold 11px -apple-system, BlinkMacSystemFont, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(label, pos.x, pos.y + nodeRadius + 16);

    // Segment count subtitle
    ctx.fillStyle = "#94a3b8";
    ctx.font = "9px -apple-system, BlinkMacSystemFont, sans-serif";
    ctx.fillText(`${pos.weight} segs`, pos.x, pos.y + nodeRadius + 30);
  });

  // Draw legend
  const legendY = H - 45;

  // Red legend (merge candidates)
  ctx.fillStyle = "rgba(239, 68, 68, 0.5)";
  ctx.fillRect(20, legendY, 22, 3);
  ctx.fillStyle = "#94a3b8";
  ctx.font = "11px -apple-system, BlinkMacSystemFont, sans-serif";
  ctx.textAlign = "left";
  ctx.fillText("≥85% similarity (merge candidate)", 50, legendY + 5);

  // Gray legend (separate)
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(20, legendY + 20);
  ctx.lineTo(42, legendY + 20);
  ctx.strokeStyle = "rgba(148, 163, 184, 0.4)";
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillText("<85% similarity (separate speakers)", 50, legendY + 25);

  // Render histogram
  _renderSimDistribution(labels, simMatrix);

  console.log(
    `[Network] Rendered ${labels.length} nodes, ${edges.length} edges`,
  );
}

/**
 * Destroy all network charts
 */
function destroyNetworkCharts() {
  Object.keys(networkCharts).forEach((key) => {
    if (networkCharts[key]) {
      networkCharts[key].destroy();
      networkCharts[key] = null;
    }
  });
  console.log("[Network] All charts destroyed");
}

// ===== PRIVATE FUNCTIONS =====

/**
 * Render similarity distribution histogram
 * @private
 */
function _renderSimDistribution(labels, simMatrix) {
  const wrapper = document.getElementById("simDistWrapper");
  if (!wrapper) return;

  wrapper.innerHTML = '<canvas id="simDistChart"></canvas>';
  const canvas = document.getElementById("simDistChart");
  if (!canvas) return;

  const ctx = canvas.getContext("2d");

  // Collect all pairwise similarities
  const allSims = [];
  for (let i = 0; i < labels.length; i++) {
    for (let j = i + 1; j < labels.length; j++) {
      const sim = simMatrix[i]?.[j];
      if (sim !== undefined && sim !== null) {
        allSims.push(sim);
      }
    }
  }

  if (allSims.length === 0) {
    wrapper.innerHTML =
      '<div class="empty-state"><p>No similarity data</p></div>';
    return;
  }

  // Create histogram bins
  const bins = 10;
  const histogram = new Array(bins).fill(0);
  const binLabels = [];

  for (let i = 0; i < bins; i++) {
    const start = i / bins;
    const end = (i + 1) / bins;
    binLabels.push(`${(start * 100).toFixed(0)}-${(end * 100).toFixed(0)}%`);

    allSims.forEach((sim) => {
      if (sim >= start && (sim < end || (i === bins - 1 && sim <= end))) {
        histogram[i]++;
      }
    });
  }

  // Color bins by threshold zones
  const binColors = binLabels.map((_, i) => {
    const midPoint = (i + 0.5) / bins;
    if (midPoint >= 0.85) return "#ef4444CC"; // Red - merge zone
    if (midPoint >= 0.5) return "#eab308CC"; // Yellow - borderline
    return "#22c55eCC"; // Green - well separated
  });

  if (networkCharts.simDist) networkCharts.simDist.destroy();

  networkCharts.simDist = new Chart(ctx, {
    type: "bar",
    data: {
      labels: binLabels,
      datasets: [
        {
          label: "Number of Pairs",
          data: histogram,
          backgroundColor: binColors,
          borderColor: binColors.map((c) => c.replace("CC", "")),
          borderWidth: 1,
          borderRadius: 4,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.raw} speaker pair(s)`,
            afterLabel: (ctx) => {
              const midPoint = (ctx.dataIndex + 0.5) / bins;
              if (midPoint >= 0.85) return "⚠️ Above merge threshold";
              if (midPoint >= 0.5) return "🔶 Borderline similarity";
              return "✅ Well separated";
            },
          },
        },
        title: {
          display: true,
          text: "Pairwise Similarity Distribution",
          color: "#94a3b8",
          font: { size: 13 },
        },
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { color: "#94a3b8", font: { size: 10 } },
          title: {
            display: true,
            text: "Cosine Similarity Range",
            color: "#94a3b8",
          },
        },
        y: {
          grid: { color: "rgba(148,163,184,0.06)" },
          ticks: { color: "#64748b", stepSize: 1 },
          title: { display: true, text: "Number of Pairs", color: "#94a3b8" },
          beginAtZero: true,
        },
      },
    },
  });

  console.log(`[Network] Rendered distribution with ${allSims.length} pairs`);
}

/**
 * Show empty state for distribution chart
 * @private
 */
function _renderEmptySimDist() {
  const wrapper = document.getElementById("simDistWrapper");
  if (wrapper) {
    wrapper.innerHTML =
      '<div class="empty-state"><p>No similarity data available</p></div>';
  }
}
