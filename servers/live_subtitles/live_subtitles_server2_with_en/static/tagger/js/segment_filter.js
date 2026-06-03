// ===== Segment Count Filter =====
function setSegmentCount(count, btn) {
  currentSegmentCount = count;
  document
    .querySelectorAll("#presetButtons .btn")
    .forEach((b) => b.classList.remove("active"));
  if (btn) btn.classList.add("active");
  const ci = document.getElementById("customSegmentCount");
  ci.value = "";
  ci.classList.remove("active");
  updateSegmentInfo();
  refreshPlots();
}
function applyCustomSegmentCount() {
  const ci = document.getElementById("customSegmentCount");
  const raw = ci.value.trim();
  if (!raw) return;
  const count = parseInt(raw, 10);
  if (isNaN(count) || count < 1) {
    ci.style.borderColor = "#f44336";
    setTimeout(() => (ci.style.borderColor = "#ddd"), 1500);
    return;
  }
  const capped = Math.min(count, 10000);
  currentSegmentCount = capped;
  ci.value = capped;
  ci.classList.add("active");
  document
    .querySelectorAll("#presetButtons .btn")
    .forEach((b) => b.classList.remove("active"));
  updateSegmentInfo();
  refreshPlots();
}
function highlightCustomInput(f) {
  const ci = document.getElementById("customSegmentCount");
  if (f) ci.classList.add("active");
  else if (!ci.value.trim()) ci.classList.remove("active");
}
function updateSegmentInfo() {
  const fs = document.getElementById("filterSelect");
  const mode = fs ? fs.value : "all";
  const label =
    mode === "speech"
      ? " (speech only)"
      : mode === "non-speech"
        ? " (non-speech only)"
        : "";
  document.getElementById("segmentInfo").textContent =
    `Showing last ${currentSegmentCount} segments${label}`;
}
function groupChunksBySegment(chunks) {
  const map = new Map();
  chunks.forEach((chunk, idx) => {
    const sd = chunk.segment_dir || "";
    const sn = chunk.segment_number;
    let key =
      sd ||
      (sn !== undefined && sn !== null
        ? `seg_${String(sn).padStart(3, "0")}`
        : `chunk_${idx}`);
    if (!map.has(key))
      map.set(key, {
        segmentKey: key,
        segmentDir: sd,
        segmentNumber: sn,
        chunks: [],
        totalSpeechProb: 0,
        speechChunks: 0,
      });
    const seg = map.get(key);
    seg.chunks.push(chunk);
    if (chunk.speech_probability != null)
      seg.totalSpeechProb += chunk.speech_probability;
    if (chunk.speech_detected) seg.speechChunks++;
  });
  const arr = Array.from(map.values());
  arr.sort((a, b) => {
    const an = parseInt(a.segmentKey.replace(/\D/g, ""), 10),
      bn = parseInt(b.segmentKey.replace(/\D/g, ""), 10);
    return !isNaN(an) && !isNaN(bn)
      ? an - bn
      : a.segmentKey.localeCompare(b.segmentKey);
  });
  return arr;
}
function filterSegments(segments, count, speechFilter) {
  let f = [...segments];
  if (speechFilter === "speech")
    f = f.filter((s) => s.chunks.some((c) => c.speech_detected));
  else if (speechFilter === "non-speech")
    f = f.filter((s) => !s.chunks.some((c) => c.speech_detected));
  if (f.length > count) f = f.slice(-count);
  return f;
}
function flattenSegmentsToChunks(segments) {
  const all = [];
  segments.forEach((seg) => {
    seg.chunks.forEach((chunk) => {
      all.push({
        ...chunk,
        _segmentKey: seg.segmentKey,
        _segmentDir: seg.segmentDir,
      });
    });
  });
  return all;
}
