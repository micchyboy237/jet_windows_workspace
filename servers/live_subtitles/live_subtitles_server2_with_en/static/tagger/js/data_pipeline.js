// ===== Data Pipeline =====
async function fetchAllData() {
  const resp = await fetch("/tags/chunks?limit=500&offset=0");
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  const data = await resp.json();
  return { chunks: data.chunks || [], stats: data.stats || {} };
}
function getTopEventNames(chunks, topN) {
  const stats = {};
  chunks.forEach(chunk => {
    (chunk.top_predictions||[]).forEach(pred => {
      if (!pred.name) return;
      if (!stats[pred.name]) stats[pred.name] = { count:0, totalProb:0, maxProb:0 };
      stats[pred.name].count++; stats[pred.name].totalProb += pred.prob;
      stats[pred.name].maxProb = Math.max(stats[pred.name].maxProb, pred.prob);
    });
  });
  return Object.entries(stats).sort((a,b) => b[1].count - a[1].count).slice(0, topN)
    .map(([name, s]) => ({ name, count: s.count, avgProb: s.totalProb/s.count, maxProb: s.maxProb }));
}
function buildHeatmapData(chunks, topEventNames) {
  const matrix = [];
  topEventNames.forEach(name => {
    const row = chunks.map(chunk => { const pred = (chunk.top_predictions||[]).find(p=>p.name===name); return pred?pred.prob:0; });
    matrix.push(row);
  });
  return matrix;
}
async function fetchAndProcessData() {
  if (activeSegmentFilter) {
    const { chunks: all, stats } = await fetchAllData();
    const filtered = all.filter(c => {
      const sd = c.segment_dir||"", sn = c.segment_number;
      return sd===activeSegmentFilter || (sn?`seg_${String(sn).padStart(3,"0")}`:"")===activeSegmentFilter || String(sn)===activeSegmentFilter;
    });
    return { chunks: filtered, segments: groupChunksBySegment(filtered), stats, isFiltered: true };
  }
  const { chunks: all, stats } = await fetchAllData();
  const allSegs = groupChunksBySegment(all);
  const filterMode = document.getElementById("filterSelect")?.value || "all";
  const filteredSegs = filterSegments(allSegs, currentSegmentCount, filterMode);
  updateSegmentInfo();
  return { chunks: flattenSegmentsToChunks(filteredSegs), segments: filteredSegs, allSegments: allSegs, stats, isFiltered: false };
}