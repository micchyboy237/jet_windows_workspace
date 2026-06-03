// ===== Chunks Summary Grid =====
function createChunksSummary(chunks, topN) {
  const container = document.createElement("div");
  container.className = "plot-container xlarge";
  container.style.cssText =
    "display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:10px;overflow-y:auto;";
  const MAX = 24;
  chunks.slice(0, MAX).forEach((chunk, idx) => {
    const card = document.createElement("div");
    card.style.cssText =
      "border:1px solid #DDD;border-radius:5px;padding:8px;background:#FAFAFA;position:relative;min-height:150px;";
    const preds = (chunk.top_predictions || []).slice(0, topN);
    const maxP = preds.length > 0 ? Math.max(...preds.map((p) => p.prob)) : 0;
    if (maxP >= HIGH_PROBABILITY_THRESHOLD) {
      card.style.borderColor = "#FFD700";
      card.style.borderWidth = "2px";
      card.style.boxShadow = "0 0 6px rgba(255,215,0,.3)";
    } else if (maxP >= MEDIUM_PROBABILITY_THRESHOLD) {
      card.style.borderColor = "#648FFF";
      card.style.borderWidth = "1.5px";
    } else if (maxP >= DEFAULT_PROBABILITY_THRESHOLD) {
      card.style.borderColor = "#888";
    } else {
      card.style.borderColor = "#DDD";
    }
    const title = document.createElement("div");
    title.style.cssText =
      "font-weight:bold;font-size:10px;margin-bottom:5px;color:#333;";
    title.textContent = `C${chunk.chunk_index || idx + 1}: ${(chunk.start_time || 0).toFixed(1)}-${(chunk.end_time || 0).toFixed(1)}s`;
    card.appendChild(title);
    if (preds.length === 0) {
      const nd = document.createElement("div");
      nd.textContent = "No predictions";
      nd.style.cssText =
        "color:#999;font-size:10px;text-align:center;padding:20px;";
      card.appendChild(nd);
    } else {
      const mc = document.createElement("canvas");
      mc.style.cssText = "width:100%;height:120px;";
      card.appendChild(mc);

      const rp = [...preds].reverse();
      // Safely generate the Chart.js config
      const chartLabels = rp.map((p) =>
        p.name && p.name.length > 25
          ? p.name.substring(0, 25) + "\u2026"
          : p.name,
      );
      const chartData = rp.map((p) => p.prob);
      const barBgColors = rp.map((p) => getBarStyle(p.prob).color);
      const barBorderColors = rp.map((p) => getBarStyle(p.prob).borderColor);
      const barBorderWidths = rp.map((p) => getBarStyle(p.prob).borderWidth);

      new Chart(mc.getContext("2d"), {
        type: "bar",
        data: {
          labels: chartLabels,
          datasets: [
            {
              data: chartData,
              backgroundColor: barBgColors,
              borderColor: barBorderColors,
              borderWidth: barBorderWidths,
              borderRadius: 2,
            },
          ],
        },
        options: {
          indexAxis: "y",
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                label: function (ctx) {
                  return `${(ctx.parsed.x * 100).toFixed(1)}%`;
                },
              },
            },
          },
          scales: {
            x: {
              min: 0,
              max: 1.1,
              ticks: {
                callback: function (v) {
                  if (v === 0) return "0%";
                  if (v === 1) return "100%";
                  return "";
                },
                font: { size: 7 },
              },
            },
            y: {
              ticks: {
                font: { size: 7 },
                autoSkip: false,
              },
            },
          },
        },
      });

      const tp = rp[rp.length - 1].prob;
      const lbl = document.createElement("div");
      lbl.style.cssText = "position:absolute;top:30px;right:5px;font-size:9px;";
      lbl.style.fontWeight =
        tp >= HIGH_PROBABILITY_THRESHOLD ? "bold" : "normal";
      lbl.style.color = tp >= HIGH_PROBABILITY_THRESHOLD ? "#1A1A6E" : "#555";
      lbl.textContent =
        (tp >= HIGH_PROBABILITY_THRESHOLD ? "\u2605 " : "") +
        (tp * 100).toFixed(0) +
        "%";
      card.appendChild(lbl);
    }
    container.appendChild(card);
  });
  if (chunks.length > MAX) {
    const n = document.createElement("div");
    n.style.cssText =
      "grid-column:1/-1;text-align:center;color:#999;font-size:12px;padding:10px;";
    n.textContent = `Showing first ${MAX} of ${chunks.length} chunks`;
    container.appendChild(n);
  }
  return container;
}
