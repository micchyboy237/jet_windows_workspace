// ===== Tags Page: Main Render & Init =====
async function refreshPlots() {
  const grid = document.getElementById("plotsGrid");
  if (!grid) {
    console.warn("plotsGrid element not found");
    return;
  }
  grid.innerHTML = '<div class="loading">Loading visualizations</div>';
  destroyCharts();
  try {
    const data = await fetchAndProcessData();
    const chunks = data.chunks || [],
      segments = data.segments || [],
      stats = data.stats || {};
    const isFiltered = data.isFiltered || false;
    const topN = parseInt(document.getElementById("topNSelect")?.value || "5");
    updateSummaryCards(stats, chunks, segments, isFiltered);
    if (isFiltered) {
      if (chunks.length === 0) {
        grid.innerHTML =
          '<div class="plot-card full-width"><div class="error-message"><h3>🔍 Segment Not Found</h3><p>The segment was not found.</p><a href="/tags" class="btn btn-small" style="margin-top:10px;">📊 View All</a></div></div>';
        return;
      }
      grid.innerHTML = "";
      grid.appendChild(createSegmentDetailCard(chunks[0]));
      const te = getTopEventNames(chunks, topN);
      if (te.length > 0) {
        const p = document.createElement("div");
        p.className = "plot-card full-width";
        p.innerHTML =
          '<h2>🔝 Top Predictions</h2><div class="plot-container"></div>';
        p.querySelector(".plot-container").appendChild(createResultsBar(te));
        grid.appendChild(p);
      }
      return;
    }
    if (chunks.length === 0) {
      grid.innerHTML =
        '<div class="plot-card full-width"><div class="info-message"><h3>📭 No Data</h3><p>No chunks found. Try adjusting filters.</p></div></div>';
      return;
    }
    grid.innerHTML = "";
    const topEvents = getTopEventNames(chunks, topN);
    if (topEvents.length === 0) {
      grid.innerHTML =
        '<div class="plot-card full-width"><div class="info-message"><h3>📊 No Predictions</h3><p>Chunks exist but have no prediction data.</p></div></div>';
      return;
    }
    const p1 = document.createElement("div");
    p1.className = "plot-card full-width";
    p1.innerHTML = `<h2>🔥 Event Probability Heatmap</h2><p style="color:#666;font-size:.85em;margin-bottom:10px;">Segments: ${segments.length} | Chunks: ${chunks.length} | Top Events: ${topN}</p>`;
    p1.appendChild(createChunkHeatmap(chunks, topEvents));
    grid.appendChild(p1);
    const p2 = document.createElement("div");
    p2.className = "plot-card full-width";
    p2.innerHTML =
      '<h2>📈 Event Probabilities Over Time</h2><p style="color:#666;font-size:.85em;margin-bottom:10px;">Marker size ∝ probability</p><div class="plot-container large"></div>';
    p2.querySelector(".plot-container").appendChild(
      createEventsTimeline(chunks, topEvents),
    );
    grid.appendChild(p2);
    const p3 = document.createElement("div");
    p3.className = "plot-card";
    p3.innerHTML =
      '<h2>📊 Aggregated Results</h2><p style="color:#666;font-size:.85em;margin-bottom:10px;">★ = High confidence</p><div class="plot-container"></div>';
    p3.querySelector(".plot-container").appendChild(
      createResultsBar(topEvents),
    );
    grid.appendChild(p3);
    const p4 = document.createElement("div");
    p4.className = "plot-card full-width";
    p4.innerHTML = `<h2>📋 Per-Chunk Top-${Math.min(topN, 3)} Predictions</h2><p style="color:#666;font-size:.85em;margin-bottom:10px;">Border color = confidence</p>`;
    p4.appendChild(createChunksSummary(chunks, Math.min(topN, 3)));
    grid.appendChild(p4);
  } catch (e) {
    console.error(e);
    grid.innerHTML = `<div class="plot-card full-width"><div class="error-message"><h3>❌ Error</h3><p>${escapeHtml(e.message)}</p><button class="btn btn-small" onclick="refreshPlots()" style="margin-top:10px;">🔄 Retry</button></div></div>`;
  }
}

// Initialize when DOM is ready
(function () {
  // Check if this is the tags analytics page (has plotsGrid)
  if (document.getElementById("plotsGrid")) {
    document.addEventListener("DOMContentLoaded", () => {
      // Ensure required functions exist before calling
      if (typeof applySegmentFilterFromUrl === "function") {
        applySegmentFilterFromUrl();
      } else {
        console.warn(
          "applySegmentFilterFromUrl not defined yet, skipping URL filter",
        );
      }

      if (typeof updateSegmentInfo === "function") {
        updateSegmentInfo();
      } else {
        console.warn(
          "updateSegmentInfo not defined yet, skipping segment info",
        );
      }

      if (typeof refreshPlots === "function") {
        refreshPlots();
      }

      setInterval(() => {
        if (!activeSegmentFilter && typeof refreshPlots === "function") {
          refreshPlots();
        }
      }, 60000);
    });
  }
})();

window.addEventListener("resize", () => {
  Object.values(charts).forEach((c) => {
    try {
      if (c && typeof c.resize === "function") {
        c.resize();
      }
    } catch (e) {
      // Silently ignore resize errors
    }
  });
});
