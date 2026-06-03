// ===== Segment Detail Card =====
function createSegmentDetailCard(chunk) {
  const card = document.createElement("div");
  card.className = "plot-card full-width";
  const sd = chunk.segment_dir || "N/A";
  const sp = chunk.speech_detected;
  const spp = chunk.speech_probability || 0;
  const pm = chunk.processing_mode || "unknown";
  const preds = chunk.top_predictions || [];
  let ph = "";
  preds.forEach(p=>{ph+=`<tr><td>${escapeHtml(p.name)}</td><td><div class="prob-bar"><div class="prob-fill" style="width:${p.prob*100}%"></div></div></td><td><strong>${(p.prob*100).toFixed(1)}%</strong></td></tr>`;});
  card.innerHTML = `<h2>📋 Chunk Detail: ${escapeHtml(sd)}</h2>
    <table class="detail-table">
      <tr><th>Segment</th><td><strong>${escapeHtml(sd)}</strong></td></tr>
      <tr><th>Start</th><td>${(chunk.start_time||0).toFixed(2)}s</td></tr>
      <tr><th>End</th><td>${(chunk.end_time||0).toFixed(2)}s</td></tr>
      <tr><th>Speech</th><td><span class="badge ${sp?"badge-success":"badge-danger"}">${sp?"✅ Yes":"❌ No"}</span></td></tr>
      <tr><th>Speech Prob</th><td><div class="prob-bar"><div class="prob-fill ${sp?"speech":""}" style="width:${spp*100}%"></div></div><small>${(spp*100).toFixed(1)}%</small></td></tr>
      <tr><th>Mode</th><td><span class="badge badge-info">${escapeHtml(pm)}</span></td></tr>
    </table>
    ${preds.length>0?`<h3 style="margin-top:15px;color:#667eea;">🔝 Top Predictions</h3><table class="detail-table"><thead><tr><th>Label</th><th>Prob Bar</th><th>Value</th></tr></thead><tbody>${ph}</tbody></table>`:'<p style="margin-top:15px;color:#666;">No predictions</p>'}`;
  return card;
}