// ===== Shared Chart Helpers =====

function getHeatmapColor(prob) {
  if (prob >= 0.8) return "#1A1A6E";
  if (prob >= 0.5) return "#648FFF";
  if (prob >= 0.2) return "#A8C4FF";
  if (prob > 0) return "#D6E4FF";
  return "#FFFFFF";
}

function getTextColor(prob) { return prob >= 0.5 ? "#FFFFFF" : "#333333"; }

function getMarkerSize(prob) {
  if (prob >= HIGH_PROBABILITY_THRESHOLD) return 12;
  if (prob >= MEDIUM_PROBABILITY_THRESHOLD) return 8;
  if (prob >= DEFAULT_PROBABILITY_THRESHOLD) return 6;
  return 4;
}

function getBarStyle(prob) {
  if (prob >= HIGH_PROBABILITY_THRESHOLD)
    return { color: "#1A1A6E", borderColor: "#FFD700", borderWidth: 2, alpha: 1.0 };
  if (prob >= MEDIUM_PROBABILITY_THRESHOLD)
    return { color: "#DC267F", borderColor: "#FFFFFF", borderWidth: 1, alpha: 0.9 };
  if (prob >= DEFAULT_PROBABILITY_THRESHOLD)
    return { color: "#648FFF", borderColor: "#FFFFFF", borderWidth: 1, alpha: 0.8 };
  return { color: "#AAAAAA", borderColor: "#888888", borderWidth: 1, alpha: 0.6 };
}