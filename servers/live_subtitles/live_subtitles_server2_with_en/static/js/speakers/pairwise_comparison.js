// ============================================================
// pairwise_comparison.js
// External module for Pairwise Speaker Comparison
// NOTE: The inline <script> in pairwise_comparison.html is the
// canonical source for pairwise logic. This file adds only
// supplemental functionality that references the existing
// functions and state already set up by the inline script.
//
// DO NOT redeclare pairwiseData, pairwiseCharts, or any functions
// already defined in pairwise_comparison.html.
// ============================================================

(function () {
  "use strict";

  // Guard: only run if the inline script has initialized
  if (typeof window._pairwiseInitGuard === "undefined") {
    console.warn(
      "[PairwiseJS] Inline script not initialized yet, deferring...",
    );
    // Retry after a short delay
    setTimeout(arguments.callee, 50);
    return;
  }

  console.log(
    "[PairwiseJS] External module loaded, patching into existing pairwise system",
  );

  // The inline script uses #pairwiseGauge as the gauge element ID
  // and updatePairwiseGauge(similarity) as the update function.
  // The external JS originally used #similarityGauge and _updateSimGauge.
  // We keep backward compatibility by aliasing, but the canonical
  // element ID is now #pairwiseGauge.

  // Ensure both gauge IDs work (in case some old code references #similarityGauge)
  const originalUpdateGauge =
    window.updatePairwiseGauge ||
    function (sim) {
      // Fallback: try both element IDs
      const gauge =
        document.getElementById("pairwiseGauge") ||
        document.getElementById("similarityGauge");
      if (!gauge) return;
      const pct = Math.round(Math.max(0, Math.min(1, sim)) * 100);
      let color;
      if (sim >= 0.8) color = "#ef4444";
      else if (sim >= 0.5) color = "#eab308";
      else color = "#22c55e";
      gauge.style.setProperty("--similarity-pct", pct);
      gauge.style.setProperty("--similarity-color", color);
      const span = gauge.querySelector("span");
      if (span) span.textContent = pct + "%";
    };

  // Expose a stable API that won't conflict
  window.PairwiseComparison = {
    getData: function () {
      return window._pairwiseData || null;
    },
    getCharts: function () {
      return window._pairwiseCharts || {};
    },
    updateGauge: function (similarity) {
      originalUpdateGauge(similarity);
    },
    refresh: function () {
      if (typeof window.updatePairwiseComparison === "function") {
        window.updatePairwiseComparison();
      }
    },
  };

  console.log(
    "[PairwiseJS] PairwiseComparison API exposed at window.PairwiseComparison",
  );
})();
