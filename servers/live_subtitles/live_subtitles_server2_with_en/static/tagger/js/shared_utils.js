// ===== Shared Utility Functions =====
function escapeHtml(str) {
  if (!str) return "";
  const div = document.createElement("div");
  div.appendChild(document.createTextNode(str));
  return div.innerHTML;
}

function getQueryParams() {
  const params = new URLSearchParams(window.location.search);
  return { segment: params.get("segment") };
}

function destroyCharts() {
  Object.values(charts).forEach((chart) => {
    try {
      chart.destroy();
    } catch (e) {}
  });
  charts = {};
}

function applySegmentFilterFromUrl() {
  const { segment } = getQueryParams();
  activeSegmentFilter = segment;
  const banner = document.getElementById("filterBanner");
  const controls = document.getElementById("globalControls");
  const navHome = document.getElementById("navHome");
  if (activeSegmentFilter) {
    if (banner) banner.classList.remove("hidden");
    if (document.getElementById("filterBannerSegmentName")) {
      document.getElementById("filterBannerSegmentName").textContent =
        activeSegmentFilter;
    }
    if (controls) controls.style.display = "none";
    if (navHome) {
      navHome.href = "/tags";
      navHome.textContent = "📊 All Segments";
    }
  } else {
    if (banner) banner.classList.add("hidden");
    if (controls) controls.style.display = "flex";
    if (navHome) {
      navHome.href = "/tags";
      navHome.textContent = "📊 Home";
    }
  }
}
