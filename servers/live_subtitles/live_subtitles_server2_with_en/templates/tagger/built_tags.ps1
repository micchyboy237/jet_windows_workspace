<#
.SYNOPSIS
    Builds tags.html and dashboard.html from modular fragments.
.DESCRIPTION
    Reads layout templates and inlines CSS/JS fragments.
    Shared fragments (CSS, constants, utils, chart helpers) are used by both pages.
    Auto-creates missing files on first run.
    Run this from the templates\tagger directory.
.NOTES
    Usage: .\build_tags.ps1
    Output: tags.html, dashboard.html (overwritten)
#>

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Building tags.html & dashboard.html" -ForegroundColor Cyan
Write-Host "  from modular fragments" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# ---- Paths ----
$TagsLayoutFile      = Join-Path $ScriptDir "tags_layout.html"
$DashboardLayoutFile = Join-Path $ScriptDir "dashboard_layout.html"
$FragmentsDir        = Join-Path $ScriptDir "fragments"
$CssDir              = Join-Path $FragmentsDir "css"
$JsDir               = Join-Path $FragmentsDir "js"
$TagsOutputFile      = Join-Path $ScriptDir "tags.html"
$DashboardOutputFile = Join-Path $ScriptDir "dashboard.html"

# ---- Helper: Ensure a directory exists ----
function Ensure-Directory($path) {
    if (-not (Test-Path $path)) {
        Write-Host "  + Creating directory: $path" -ForegroundColor Green
        New-Item -ItemType Directory -Path $path -Force | Out-Null
    }
}

# ---- Helper: Create a fragment file if missing ----
function Ensure-Fragment($relativePath, $description, $defaultContent) {
    $fullPath = Join-Path $ScriptDir $relativePath
    $parentDir = Split-Path $fullPath -Parent
    Ensure-Directory $parentDir

    if (Test-Path $fullPath) {
        Write-Host "  ✓ Already exists: $relativePath" -ForegroundColor Gray
        return
    }
    Write-Host "  + Creating: $relativePath ($description)" -ForegroundColor Green
    Set-Content -Path $fullPath -Value $defaultContent -Encoding UTF8 -NoNewline
}

# ---- Helper: Create a layout file if missing ----
function Ensure-Layout($filePath, $description, $defaultContent) {
    if (Test-Path $filePath) {
        Write-Host "  ✓ Already exists: $(Split-Path $filePath -Leaf)" -ForegroundColor Gray
        return
    }
    Write-Host "  + Creating: $(Split-Path $filePath -Leaf) ($description)" -ForegroundColor Green
    Set-Content -Path $filePath -Value $defaultContent -Encoding UTF8 -NoNewline
}

# ---- Helper: Read fragment safely ----
function Read-Fragment($relativePath) {
    $fullPath = Join-Path $ScriptDir $relativePath
    if (-not (Test-Path $fullPath)) {
        Write-Warning "Fragment not found, using empty placeholder: $relativePath"
        return "/* MISSING: $relativePath */"
    }
    $content = Get-Content $fullPath -Raw -Encoding UTF8
    Write-Host "  ✓ Loaded: $relativePath ($($content.Length) chars)" -ForegroundColor Green
    return $content
}

# ---- Helper: Build a single HTML file from layout + fragments ----
function Build-HtmlFile($layoutPath, $outputPath, $pageName) {
    Write-Host ""
    Write-Host "--- Building $pageName ---" -ForegroundColor Yellow
    Write-Host ""

    $layoutContent = Get-Content $layoutPath -Raw -Encoding UTF8
    Write-Host "  Layout loaded ($($layoutContent.Length) chars)" -ForegroundColor Gray

    # Load shared fragments
    $sharedCss         = Read-Fragment "fragments\css\shared_css.css"
    $sharedConstantsJs = Read-Fragment "fragments\js\shared_constants.js"
    $sharedUtilsJs     = Read-Fragment "fragments\js\shared_utils.js"
    $sharedChartsJs    = Read-Fragment "fragments\js\shared_charts.js"

    # Load page-specific fragments (may be empty if not used by this page)
    $segmentFilterJs   = Read-Fragment "fragments\js\segment_filter.js"
    $dataPipelineJs    = Read-Fragment "fragments\js\data_pipeline.js"
    $summaryCardsJs    = Read-Fragment "fragments\js\summary_cards.js"
    $heatmapChartJs    = Read-Fragment "fragments\js\heatmap_chart.js"
    $timelineChartJs   = Read-Fragment "fragments\js\timeline_chart.js"
    $resultsBarJs      = Read-Fragment "fragments\js\results_bar.js"
    $chunksSummaryJs   = Read-Fragment "fragments\js\chunks_summary.js"
    $detailCardJs      = Read-Fragment "fragments\js\detail_card.js"
    $mainAppJs         = Read-Fragment "fragments\js\main_app.js"
    $dashboardMainJs   = Read-Fragment "fragments\js\dashboard_main.js"

    Write-Host ""
    Write-Host "  Assembling..." -ForegroundColor Yellow

    # Replace shared placeholders
    $layoutContent = $layoutContent.Replace("<!-- INLINE_CSS -->", $sharedCss)
    $layoutContent = $layoutContent.Replace("<!-- INLINE_JS_CONSTANTS -->", $sharedConstantsJs)
    $layoutContent = $layoutContent.Replace("<!-- INLINE_JS_UTILS -->", $sharedUtilsJs)
    $layoutContent = $layoutContent.Replace("<!-- INLINE_JS_CHARTS -->", $sharedChartsJs)

    # Replace page-specific placeholders
    $layoutContent = $layoutContent.Replace("<!-- INLINE_JS_SEGMENT_FILTER -->", $segmentFilterJs)
    $layoutContent = $layoutContent.Replace("<!-- INLINE_JS_DATA_PIPELINE -->", $dataPipelineJs)
    $layoutContent = $layoutContent.Replace("<!-- INLINE_JS_SUMMARY_CARDS -->", $summaryCardsJs)
    $layoutContent = $layoutContent.Replace("<!-- INLINE_JS_HEATMAP -->", $heatmapChartJs)
    $layoutContent = $layoutContent.Replace("<!-- INLINE_JS_TIMELINE -->", $timelineChartJs)
    $layoutContent = $layoutContent.Replace("<!-- INLINE_JS_RESULTS_BAR -->", $resultsBarJs)
    $layoutContent = $layoutContent.Replace("<!-- INLINE_JS_CHUNKS_SUMMARY -->", $chunksSummaryJs)
    $layoutContent = $layoutContent.Replace("<!-- INLINE_JS_DETAIL_CARD -->", $detailCardJs)
    $layoutContent = $layoutContent.Replace("<!-- INLINE_JS_MAIN_APP -->", $mainAppJs)
    $layoutContent = $layoutContent.Replace("<!-- INLINE_JS_DASHBOARD_MAIN -->", $dashboardMainJs)

    # Write output
    Set-Content -Path $outputPath -Value $layoutContent -Encoding UTF8 -NoNewline
    $outputSize = (Get-Item $outputPath).Length
    Write-Host "  ✓ $pageName written ($([math]::Round($outputSize/1024, 1)) KB)" -ForegroundColor Green
}

# ======================================================================
# STEP 1: Create directory structure
# ======================================================================
Write-Host "Step 1: Ensuring directory structure..." -ForegroundColor Yellow
Ensure-Directory $FragmentsDir
Ensure-Directory $CssDir
Ensure-Directory $JsDir
Write-Host ""

# ======================================================================
# STEP 2: Create layout templates if missing
# ======================================================================
Write-Host "Step 2: Checking layout templates..." -ForegroundColor Yellow

# -- tags_layout.html --
Ensure-Layout $TagsLayoutFile "Analytics page layout" @'
<!doctype html>
<html>
  <head>
    <title>Audio Tagger - Chunk Analytics Plots</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-chart-matrix@2.0.1/dist/chartjs-chart-matrix.min.js"></script>
    <style>
      <!-- INLINE_CSS -->
    </style>
  </head>
  <body>
    <div class="container">
      <div class="header">
        <h1>📊 Audio Tagger - Chunk Analytics Plots</h1>
        <p>Visual analysis of chunked audio tagging results (matching audio_tagger_chunk_plots.py)</p>
        <div class="nav-buttons">
          <a href="/tags" class="btn" id="navHome">📊 Home</a>
          <a href="/tags/dashboard" class="btn btn-secondary">📋 Dashboard</a>
          <a href="/tags/chunks" class="btn btn-secondary" target="_blank">🔌 API: Chunks</a>
        </div>
      </div>
      <div class="filter-banner hidden" id="filterBanner">
        <span class="filter-info">
          🔍 <strong>Filtering by segment:</strong>
          <span id="filterBannerSegmentName">-</span>
        </span>
        <div>
          <a href="/tags" class="btn btn-warning btn-small">✕ Clear Filter</a>
          <a href="/tags/dashboard" class="btn btn-secondary btn-small" style="margin-left: 10px">📋 Back to Dashboard</a>
        </div>
      </div>
      <div class="summary-cards" id="summaryCards">
        <div class="summary-card">
          <div class="label">Unique Segments</div>
          <div class="value" id="cardSegments">-</div>
          <div class="subtitle">Distinct audio segments</div>
        </div>
        <div class="summary-card">
          <div class="label">Total Chunks</div>
          <div class="value" id="cardTotal">-</div>
          <div class="subtitle">Across all segments</div>
        </div>
        <div class="summary-card">
          <div class="label">Speech Segments</div>
          <div class="value" id="cardSpeech">-</div>
          <div class="subtitle" id="cardSpeechPct">-</div>
        </div>
        <div class="summary-card">
          <div class="label">Avg Chunks/Seg</div>
          <div class="value" id="cardAvgChunks">-</div>
          <div class="subtitle">Per segment</div>
        </div>
        <div class="summary-card">
          <div class="label">Avg Prob</div>
          <div class="value" id="cardAvgProb">-</div>
          <div class="subtitle">Speech confidence</div>
        </div>
        <div class="summary-card">
          <div class="label">Top Event</div>
          <div class="value" id="cardTopEvent" style="font-size: 1.2em">-</div>
          <div class="subtitle" id="cardTopEventProb">-</div>
        </div>
      </div>
      <div class="controls" id="globalControls">
        <div class="segment-filter-group">
          <span class="filter-label">📅 Last N Segments:</span>
          <div class="preset-buttons" id="presetButtons">
            <button class="btn btn-outline btn-small active" data-count="10" onclick="setSegmentCount(10, this)">10</button>
            <button class="btn btn-outline btn-small" data-count="20" onclick="setSegmentCount(20, this)">20</button>
            <button class="btn btn-outline btn-small" data-count="50" onclick="setSegmentCount(50, this)">50</button>
            <button class="btn btn-outline btn-small" data-count="100" onclick="setSegmentCount(100, this)">100</button>
            <button class="btn btn-outline btn-small" data-count="200" onclick="setSegmentCount(200, this)">200</button>
          </div>
          <div class="custom-input-wrapper">
            <input type="number" class="custom-input" id="customSegmentCount" placeholder="Custom" min="1" max="10000"
              onkeydown="if(event.key==='Enter')applyCustomSegmentCount()"
              onfocus="highlightCustomInput(true)" onblur="highlightCustomInput(false)" />
            <button class="apply-custom-btn" onclick="applyCustomSegmentCount()" title="Apply custom count">Go</button>
          </div>
          <span class="segment-info" id="segmentInfo"></span>
        </div>
        <div style="display: flex; gap: 15px; align-items: center; flex-wrap: wrap;">
          <label>🎯 Filter:
            <select id="filterSelect" onchange="refreshPlots()">
              <option value="all">All Segments</option>
              <option value="speech">Speech Only</option>
              <option value="non-speech">Non-Speech Only</option>
            </select>
          </label>
          <label>🔝 Top N Events:
            <select id="topNSelect" onchange="refreshPlots()">
              <option value="3">Top 3</option>
              <option value="5" selected>Top 5</option>
              <option value="8">Top 8</option>
              <option value="10">Top 10</option>
            </select>
          </label>
          <button class="btn btn-small" onclick="refreshPlots()">🔄 Refresh Data</button>
        </div>
      </div>
      <div class="plots-grid" id="plotsGrid">
        <div class="loading">Loading visualizations</div>
      </div>
    </div>
    <script>
      <!-- INLINE_JS_CONSTANTS -->
      <!-- INLINE_JS_UTILS -->
      <!-- INLINE_JS_CHARTS -->
      <!-- INLINE_JS_SEGMENT_FILTER -->
      <!-- INLINE_JS_DATA_PIPELINE -->
      <!-- INLINE_JS_SUMMARY_CARDS -->
      <!-- INLINE_JS_HEATMAP -->
      <!-- INLINE_JS_TIMELINE -->
      <!-- INLINE_JS_RESULTS_BAR -->
      <!-- INLINE_JS_CHUNKS_SUMMARY -->
      <!-- INLINE_JS_DETAIL_CARD -->
      <!-- INLINE_JS_MAIN_APP -->
    </script>
  </body>
</html>
'@

# -- dashboard_layout.html --
Ensure-Layout $DashboardLayoutFile "Dashboard page layout" @'
<!doctype html>
<html>
  <head>
    <title>Audio Tagger Dashboard</title>
    <style>
      <!-- INLINE_CSS -->
    </style>
  </head>
  <body>
    <div class="container">
      <div class="header">
        <h1>🎵 Audio Tagger Dashboard</h1>
        <p>Analyze audio files for sound events, speech detection, and temporal patterns</p>
        <div class="nav-buttons">
          <a href="/tags" class="btn">📊 Analytics</a>
        </div>
        <p style="margin-top: 10px; font-size: 0.85em; color: #888">
          💡 <strong>Click any row</strong> in the table to view that segment's detailed analytics
        </p>
      </div>
      <div class="stats-bar">
        <div class="stat-card">
          <h3>Total Segments</h3>
          <div class="value" id="totalSegments">-</div>
        </div>
        <div class="stat-card">
          <h3>Speech Segments</h3>
          <div class="value" id="speechSegments">-</div>
        </div>
        <div class="stat-card">
          <h3>Speech %</h3>
          <div class="value" id="speechPercentage">-</div>
        </div>
        <div class="stat-card">
          <h3>Avg Speech Prob</h3>
          <div class="value" id="avgSpeechProb">-</div>
        </div>
      </div>
      <div class="grid">
        <div class="card">
          <h2>📊 Saved Audio Segments</h2>
          <div class="controls">
            <div>
              <label>Filter:
                <select id="filterSelect" onchange="loadChunks()">
                  <option value="all">All Segments</option>
                  <option value="speech">Speech Only</option>
                </select>
              </label>
              <label style="margin-left: 15px">Limit:
                <select id="limitSelect" onchange="loadChunks()">
                  <option value="20">20</option>
                  <option value="50" selected>50</option>
                  <option value="100">100</option>
                  <option value="200">200</option>
                </select>
              </label>
            </div>
            <div>
              <button class="btn btn-small" onclick="loadChunks()">🔄 Refresh</button>
            </div>
          </div>
          <div id="chunksTableContainer" class="table-container">
            <div class="loading" id="chunksLoading">Loading saved segments</div>
          </div>
        </div>
        <div>
          <div class="card">
            <h2>📋 Configuration</h2>
            <table id="configTable">
              <!-- Filled by JS -->
            </table>
          </div>
          <div class="card" style="margin-top: 20px">
            <h2>🔝 Top Predictions</h2>
            <div id="topPredictionsChart" class="chart-container">
              <div class="loading">Loading predictions</div>
            </div>
          </div>
          <div class="card" style="margin-top: 20px">
            <h2>🎤 Speech Classes</h2>
            <ul id="speechClassesList" style="list-style: none; padding: 0">
              <!-- Filled by JS -->
            </ul>
          </div>
        </div>
      </div>
      <div class="card">
        <h2>🚀 Try It Out</h2>
        <div class="upload-area" id="uploadArea">
          <h3>📁 Drop an audio file here or click to upload</h3>
          <p>Supports WAV, MP3, FLAC, and more</p>
          <input type="file" id="fileInput" accept="audio/*" style="display: none" />
        </div>
        <div style="text-align: center; margin-top: 20px">
          <label style="margin-right: 20px">
            <input type="checkbox" id="chunkedMode" /> Process in chunks
          </label>
          <label style="margin-right: 20px">
            Chunk duration: <input type="number" id="chunkDuration" value="2.0" step="0.5" min="0.5" max="30" style="width: 80px" /> s
          </label>
          <label>
            Overlap: <input type="number" id="overlapDuration" value="1.0" step="0.5" min="0.1" max="15" style="width: 80px" /> s
          </label>
        </div>
        <div style="text-align: center; margin-top: 20px">
          <button class="btn" id="tagBtn" disabled>🎯 Tag Audio</button>
          <button class="btn" id="speechBtn" disabled>🎤 Check Speech</button>
        </div>
        <div id="progress" style="display: none">
          <div class="progress-bar"><div class="fill" id="progressFill"></div></div>
          <p id="progressText" style="text-align: center; color: #666"></p>
        </div>
      </div>
      <div id="results"></div>
    </div>
    <div class="toast" id="toast"></div>
    <script>
      <!-- INLINE_JS_CONSTANTS -->
      <!-- INLINE_JS_UTILS -->
      <!-- INLINE_JS_CHARTS -->
      <!-- INLINE_JS_DASHBOARD_MAIN -->
    </script>
  </body>
</html>
'@

Write-Host ""

# ======================================================================
# STEP 3: Create fragment files if missing
# ======================================================================
Write-Host "Step 3: Checking fragment files..." -ForegroundColor Yellow
Write-Host ""

# -- SHARED CSS (used by both pages) --
Ensure-Fragment "fragments\css\shared_css.css" "Shared styles for both pages" @'
*{margin:0;padding:0;box-sizing:border-box;}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);min-height:100vh;padding:20px;}
.container{max-width:1400px;margin:0 auto;}
.header{background:white;border-radius:10px;padding:30px;margin-bottom:20px;box-shadow:0 4px 6px rgba(0,0,0,0.1);text-align:center;}
.header h1{color:#667eea;margin-bottom:10px;}
.header p{color:#666;}
.nav-buttons{display:flex;justify-content:center;gap:10px;margin-top:15px;flex-wrap:wrap;}
.btn{background:#667eea;color:white;border:none;padding:10px 25px;border-radius:5px;cursor:pointer;font-size:0.95em;text-decoration:none;transition:background 0.3s,transform 0.15s;display:inline-block;}
.btn:hover{background:#764ba2;transform:translateY(-1px);}
.btn:active{transform:translateY(0);}
.btn-secondary{background:#6c757d;}
.btn-secondary:hover{background:#5a6268;}
.btn-outline{background:transparent;color:#667eea;border:2px solid #667eea;}
.btn-outline:hover{background:#667eea;color:white;}
.btn-outline.active{background:#667eea;color:white;box-shadow:0 2px 8px rgba(102,126,234,0.4);}
.btn-warning{background:#ff9800;}
.btn-warning:hover{background:#f57c00;}
.btn-small{padding:8px 20px;font-size:0.9em;}
.filter-banner{background:linear-gradient(135deg,#667eea,#764ba2);color:white;border-radius:10px;padding:15px 20px;margin-bottom:20px;box-shadow:0 4px 6px rgba(0,0,0,0.1);display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;}
.filter-banner.hidden{display:none;}
.summary-cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:15px;margin-bottom:20px;}
.summary-card{background:white;border-radius:10px;padding:20px;box-shadow:0 4px 6px rgba(0,0,0,0.1);text-align:center;position:relative;overflow:hidden;transition:transform 0.2s;}
.summary-card:hover{transform:translateY(-2px);}
.summary-card::before{content:"";position:absolute;top:0;left:0;right:0;height:4px;background:linear-gradient(90deg,#667eea,#764ba2);}
.summary-card .label{color:#666;font-size:0.85em;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;}
.summary-card .value{color:#333;font-size:2em;font-weight:bold;}
.summary-card .subtitle{color:#999;font-size:0.8em;margin-top:5px;}
.plots-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:20px;margin-bottom:20px;}
.plot-card{background:white;border-radius:10px;padding:20px;box-shadow:0 4px 6px rgba(0,0,0,0.1);}
.plot-card.full-width{grid-column:1/-1;}
.plot-card h2{color:#667eea;margin-bottom:15px;font-size:1.1em;border-bottom:2px solid #667eea;padding-bottom:10px;}
.plot-container{position:relative;width:100%;height:400px;}
.plot-container.large{height:500px;}
.plot-container.xlarge{height:650px;}
canvas{width:100%!important;height:100%!important;}
.loading{text-align:center;padding:40px;color:#666;font-size:1.2em;}
.loading::after{content:"...";animation:dots 1.5s steps(4,end) infinite;}
@keyframes dots{0%,20%{content:".";}40%{content:"..";}60%,100%{content:"...";}}
.error-message{background:#f8d7da;color:#721c24;padding:20px;border-radius:10px;text-align:center;margin:20px 0;}
.info-message{background:#d1ecf1;color:#0c5460;padding:20px;border-radius:10px;text-align:center;margin:20px 0;}
.controls{display:flex;justify-content:space-between;align-items:center;margin-bottom:15px;gap:15px;flex-wrap:wrap;background:white;border-radius:10px;padding:15px 20px;box-shadow:0 4px 6px rgba(0,0,0,0.1);}
.controls label{font-size:0.9em;color:#666;font-weight:500;}
.controls select{padding:8px 15px;border:1px solid #ddd;border-radius:5px;font-size:0.9em;background:white;}
.segment-filter-group{display:flex;align-items:center;gap:8px;flex-wrap:wrap;}
.segment-filter-group .filter-label{font-size:0.85em;color:#666;font-weight:600;white-space:nowrap;}
.segment-filter-group .preset-buttons{display:flex;gap:4px;}
.segment-filter-group .preset-buttons .btn{padding:6px 14px;font-size:0.8em;border-radius:4px;}
.segment-filter-group .custom-input-wrapper{display:flex;align-items:center;gap:4px;position:relative;}
.segment-filter-group .custom-input{width:70px;padding:6px 10px;border:2px solid #ddd;border-radius:4px;font-size:0.85em;text-align:center;transition:border-color 0.2s;}
.segment-filter-group .custom-input:focus{outline:none;border-color:#667eea;box-shadow:0 0 0 2px rgba(102,126,234,0.2);}
.segment-filter-group .custom-input.active{border-color:#667eea;background:#f8f9ff;}
.segment-filter-group .apply-custom-btn{padding:6px 12px;font-size:0.8em;background:#667eea;color:white;border:none;border-radius:4px;cursor:pointer;transition:background 0.2s;}
.segment-filter-group .apply-custom-btn:hover{background:#764ba2;}
.segment-filter-group .segment-info{font-size:0.8em;color:#999;white-space:nowrap;margin-left:8px;}
.detail-table{width:100%;border-collapse:collapse;margin-top:10px;}
.detail-table th,.detail-table td{padding:10px;text-align:left;border-bottom:1px solid #ddd;}
.detail-table th{background:#f5f5f5;font-weight:bold;width:35%;}
.badge{display:inline-block;padding:3px 8px;border-radius:3px;font-size:0.8em;font-weight:bold;}
.badge-success{background:#d4edda;color:#155724;}
.badge-danger{background:#f8d7da;color:#721c24;}
.badge-info{background:#d1ecf1;color:#0c5460;}
.prob-bar{height:10px;background:#f0f0f0;border-radius:5px;overflow:hidden;margin-top:5px;}
.prob-fill{height:100%;background:linear-gradient(90deg,#667eea,#764ba2);border-radius:5px;}
.prob-fill.speech{background:linear-gradient(90deg,#4caf50,#45a049);}
.heatmap-legend{display:flex;align-items:center;justify-content:center;gap:10px;margin-top:10px;font-size:0.85em;}
.heatmap-gradient{width:200px;height:15px;background:linear-gradient(to right,#FFFFFF,#648FFF,#1A1A6E);border-radius:3px;border:1px solid #ddd;}

/* Dashboard-specific styles */
.stats-bar{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:15px;margin-bottom:20px;}
.stat-card{background:white;border-radius:10px;padding:20px;box-shadow:0 4px 6px rgba(0,0,0,0.1);text-align:center;}
.stat-card h3{color:#667eea;font-size:0.9em;margin-bottom:10px;text-transform:uppercase;letter-spacing:1px;}
.stat-card .value{color:#764ba2;font-size:2em;font-weight:bold;}
.grid{display:grid;grid-template-columns:2fr 1fr;gap:20px;margin-bottom:20px;}
.card{background:white;border-radius:10px;padding:20px;box-shadow:0 4px 6px rgba(0,0,0,0.1);}
.card h2{color:#667eea;margin-bottom:15px;font-size:1.2em;border-bottom:2px solid #667eea;padding-bottom:10px;}
.upload-area{border:2px dashed #667eea;border-radius:10px;padding:40px;text-align:center;background:rgba(102,126,234,0.05);margin-bottom:20px;cursor:pointer;transition:all 0.3s;}
.upload-area:hover{background:rgba(102,126,234,0.1);border-color:#764ba2;}
.upload-area.dragover{background:rgba(102,126,234,0.2);border-color:#764ba2;}
.btn:disabled{background:#ccc;cursor:not-allowed;}
.progress-bar{width:100%;height:20px;background:#f0f0f0;border-radius:10px;overflow:hidden;margin:10px 0;}
.progress-bar .fill{height:100%;background:linear-gradient(90deg,#667eea,#764ba2);width:0%;transition:width 0.3s;}
#results{background:white;border-radius:10px;padding:20px;margin-top:20px;box-shadow:0 4px 6px rgba(0,0,0,0.1);display:none;}
.result-item{background:#f5f5f5;padding:10px;border-radius:5px;margin-bottom:10px;}
table{width:100%;border-collapse:collapse;}
th,td{padding:10px;text-align:left;border-bottom:1px solid #ddd;}
th{background:#f5f5f5;font-weight:bold;position:sticky;top:0;}
tbody tr{cursor:pointer;transition:background-color 0.2s,transform 0.1s;}
tbody tr:hover{background-color:#f0f0ff;transform:scale(1.002);}
tbody tr:active{background-color:#e0e0ff;}
tbody tr.selected{background-color:#d0d0ff;outline:2px solid #667eea;}
.view-indicator{display:inline-block;margin-left:5px;color:#667eea;font-size:0.8em;opacity:0;transition:opacity 0.2s;}
tbody tr:hover .view-indicator{opacity:1;}
.table-container{max-height:500px;overflow-y:auto;margin-top:10px;}
.chart-container{margin-top:20px;padding:15px;background:#f9f9f9;border-radius:5px;}
.chart-bar{display:flex;align-items:center;margin-bottom:8px;}
.chart-label{width:150px;font-size:0.9em;color:#666;text-align:right;padding-right:10px;}
.chart-track{flex:1;height:20px;background:#e0e0e0;border-radius:10px;overflow:hidden;}
.chart-fill{height:100%;background:linear-gradient(90deg,#667eea,#764ba2);border-radius:10px;transition:width 0.5s;}
.chart-value{width:80px;font-size:0.9em;color:#333;padding-left:10px;}
.toast{position:fixed;bottom:20px;right:20px;background:#333;color:white;padding:12px 20px;border-radius:5px;opacity:0;transition:opacity 0.3s;z-index:1000;pointer-events:none;}
.toast.show{opacity:1;}

@media(max-width:768px){
  .plots-grid{grid-template-columns:1fr;}
  .summary-cards{grid-template-columns:repeat(2,1fr);}
  .controls{flex-direction:column;align-items:stretch;}
  .segment-filter-group{flex-direction:column;align-items:flex-start;}
  .grid{grid-template-columns:1fr;}
  .stats-bar{grid-template-columns:repeat(2,1fr);}
}
'@

# -- SHARED JS FRAGMENTS (used by both pages) --
Ensure-Fragment "fragments\js\shared_constants.js" "Global state and constants" @'
// ===== Global State & Constants =====
// tags.html state
let charts = {};
let activeSegmentFilter = null;
let currentSegmentCount = 10;

// Color palette matching Python COLORS
const EVENT_COLORS = [
  "#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000",
  "#009E73", "#56B4E9", "#E69F00", "#F0E442", "#0072B2"
];

// Thresholds matching Python constants
const HIGH_PROBABILITY_THRESHOLD = 0.7;
const MEDIUM_PROBABILITY_THRESHOLD = 0.4;
const DEFAULT_PROBABILITY_THRESHOLD = 0.3;

// Dashboard state
let selectedFile = null;

// Speech classes (fetched from config)
const SPEECH_CLASS_NAMES = [
  "Speech", "Speech synthesizer", "Narration, monologue",
  "Conversation", "Dialogue", "Babbling", "Children shouting",
  "Shout", "Whispering", "Laughter", "Crying", "Sighing",
  "Singing", "Humming", "Mantra", "Rapping", "Yodeling",
  "Chant", "Crowd", "Cheering", "Applause"
];
'@

Ensure-Fragment "fragments\js\shared_utils.js" "Shared utility functions" @'
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
  Object.values(charts).forEach(chart => {
    try { chart.destroy(); } catch(e) {}
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
    banner.classList.remove("hidden");
    document.getElementById("filterBannerSegmentName").textContent = activeSegmentFilter;
    controls.style.display = "none";
    navHome.href = "/tags"; navHome.textContent = "📊 All Segments";
  } else {
    banner.classList.add("hidden");
    if (controls) controls.style.display = "flex";
    navHome.href = "/tags"; navHome.textContent = "📊 Home";
  }
}
'@

Ensure-Fragment "fragments\js\shared_charts.js" "Chart style/color helpers" @'
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
'@

# -- TAGS-SPECIFIC JS FRAGMENTS --
Ensure-Fragment "fragments\js\segment_filter.js" "Segment count filter logic" @'
// ===== Segment Count Filter =====
function setSegmentCount(count, btn) {
  currentSegmentCount = count;
  document.querySelectorAll("#presetButtons .btn").forEach(b => b.classList.remove("active"));
  if (btn) btn.classList.add("active");
  const ci = document.getElementById("customSegmentCount"); ci.value = ""; ci.classList.remove("active");
  updateSegmentInfo(); refreshPlots();
}
function applyCustomSegmentCount() {
  const ci = document.getElementById("customSegmentCount");
  const raw = ci.value.trim();
  if (!raw) return;
  const count = parseInt(raw, 10);
  if (isNaN(count) || count < 1) { ci.style.borderColor="#f44336"; setTimeout(()=>ci.style.borderColor="#ddd",1500); return; }
  const capped = Math.min(count, 10000);
  currentSegmentCount = capped;
  ci.value = capped; ci.classList.add("active");
  document.querySelectorAll("#presetButtons .btn").forEach(b => b.classList.remove("active"));
  updateSegmentInfo(); refreshPlots();
}
function highlightCustomInput(f) {
  const ci = document.getElementById("customSegmentCount");
  if(f) ci.classList.add("active"); else if(!ci.value.trim()) ci.classList.remove("active");
}
function updateSegmentInfo() {
  const fs = document.getElementById("filterSelect");
  const mode = fs ? fs.value : "all";
  const label = mode==="speech"?" (speech only)":mode==="non-speech"?" (non-speech only)":"";
  document.getElementById("segmentInfo").textContent = `Showing last ${currentSegmentCount} segments${label}`;
}
function groupChunksBySegment(chunks) {
  const map = new Map();
  chunks.forEach((chunk, idx) => {
    const sd = chunk.segment_dir || "";
    const sn = chunk.segment_number;
    let key = sd || (sn !== undefined && sn !== null ? `seg_${String(sn).padStart(3,"0")}` : `chunk_${idx}`);
    if (!map.has(key)) map.set(key, { segmentKey: key, segmentDir: sd, segmentNumber: sn, chunks: [], totalSpeechProb:0, speechChunks:0 });
    const seg = map.get(key);
    seg.chunks.push(chunk);
    if (chunk.speech_probability != null) seg.totalSpeechProb += chunk.speech_probability;
    if (chunk.speech_detected) seg.speechChunks++;
  });
  const arr = Array.from(map.values());
  arr.sort((a,b) => { const an=parseInt(a.segmentKey.replace(/\D/g,""),10), bn=parseInt(b.segmentKey.replace(/\D/g,""),10); return !isNaN(an)&&!isNaN(bn)?an-bn:a.segmentKey.localeCompare(b.segmentKey); });
  return arr;
}
function filterSegments(segments, count, speechFilter) {
  let f = [...segments];
  if (speechFilter==="speech") f = f.filter(s => s.chunks.some(c => c.speech_detected));
  else if (speechFilter==="non-speech") f = f.filter(s => !s.chunks.some(c => c.speech_detected));
  if (f.length > count) f = f.slice(-count);
  return f;
}
function flattenSegmentsToChunks(segments) {
  const all = [];
  segments.forEach(seg => { seg.chunks.forEach(chunk => { all.push({...chunk, _segmentKey: seg.segmentKey, _segmentDir: seg.segmentDir}); }); });
  return all;
}
'@

Ensure-Fragment "fragments\js\data_pipeline.js" "Data fetching and processing" @'
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
'@

Ensure-Fragment "fragments\js\summary_cards.js" "Summary cards update" @'
// ===== Summary Cards =====
function updateSummaryCards(stats, chunks, segments, isFiltered) {
  if (isFiltered && chunks.length===1) {
    const s = chunks[0];
    document.getElementById("cardSegments").textContent="1";
    document.getElementById("cardTotal").textContent="1";
    document.getElementById("cardSpeech").textContent=s.speech_detected?"✅ Yes":"❌ No";
    document.getElementById("cardSpeechPct").textContent="Single chunk view";
    document.getElementById("cardAvgChunks").textContent="1";
    document.getElementById("cardAvgProb").textContent=(s.speech_probability||0).toFixed(3);
    if (s.top_predictions&&s.top_predictions.length>0) {
      document.getElementById("cardTopEvent").textContent=s.top_predictions[0].name;
      document.getElementById("cardTopEventProb").textContent=`${(s.top_predictions[0].prob*100).toFixed(1)}% probability`;
    } else { document.getElementById("cardTopEvent").textContent="N/A"; document.getElementById("cardTopEventProb").textContent="No predictions"; }
    return;
  }
  const sc = segments?segments.length:0, cc = chunks.length;
  document.getElementById("cardSegments").textContent=sc;
  document.getElementById("cardTotal").textContent=cc;
  const speechSegs = segments?segments.filter(s=>s.chunks.some(c=>c.speech_detected)).length:chunks.filter(c=>c.speech_detected).length;
  document.getElementById("cardSpeech").textContent=speechSegs;
  document.getElementById("cardSpeechPct").textContent=sc>0?`${((speechSegs/sc)*100).toFixed(1)}% of segments`:"-";
  document.getElementById("cardAvgChunks").textContent=sc>0?(cc/sc).toFixed(1):"-";
  const probs = chunks.map(c=>c.speech_probability).filter(p=>p!=null);
  document.getElementById("cardAvgProb").textContent=probs.length>0?(probs.reduce((a,b)=>a+b,0)/probs.length).toFixed(3):"-";
  const te = getTopEventNames(chunks,1);
  if (te.length>0) {
    document.getElementById("cardTopEvent").textContent=te[0].name;
    document.getElementById("cardTopEventProb").textContent=`${(te[0].avgProb*100).toFixed(1)}% avg (${te[0].count} chunks)`;
  } else { document.getElementById("cardTopEvent").textContent="N/A"; document.getElementById("cardTopEventProb").textContent="No data"; }
}
'@

Ensure-Fragment "fragments\js\heatmap_chart.js" "Heatmap chart" @'
// ===== Chunk Heatmap =====
function createChunkHeatmap(chunks, topEventNames) {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  const container = document.createElement("div");
  container.className = "plot-container xlarge";
  const nEvents = topEventNames.length, nChunks = chunks.length;
  const matrix = buildHeatmapData(chunks, topEventNames.map(e=>e.name));
  const datasets = [];
  for (let r=0; r<nEvents; r++) for (let c=0; c<nChunks; c++) datasets.push({x:c, y:r, v:matrix[r][c]});
  charts.heatmap = new Chart(ctx, {
    type:"matrix",
    data:{datasets:[{label:"Probability",data:datasets,backgroundColor(ctx){return getHeatmapColor(ctx.dataset.data[ctx.dataIndex].v);},borderColor:"#DDD",borderWidth:1,width:({chart})=>(chart.chartArea||{}).width/Math.max(nChunks,1)-1,height:({chart})=>(chart.chartArea||{}).height/Math.max(nEvents,1)-1}]},
    options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false},tooltip:{callbacks:{title:(items)=>{const ci=items[0].dataset.data[items[0].dataIndex].x,ch=chunks[ci];return`Chunk ${ch.chunk_index||ci}: ${(ch.start_time||0).toFixed(1)}s-${(ch.end_time||0).toFixed(1)}s`;},label:(item)=>{const ei=item.dataset.data[item.dataIndex].y;return`${topEventNames[ei].name}: ${(item.dataset.data[item.dataIndex].v*100).toFixed(1)}%`;}}}},scales:{x:{type:"linear",offset:true,ticks:{stepSize:1,callback:(v)=>{const i=Math.round(v);if(i>=0&&i<nChunks)return`${(chunks[i].start_time||0).toFixed(1)}s`;return"";},font:{size:8},maxRotation:60},title:{display:true,text:"Chunk Start Time"},grid:{display:false}},y:{type:"linear",offset:true,ticks:{stepSize:1,callback:(v)=>{const i=Math.round(v);if(i>=0&&i<nEvents){const n=topEventNames[i].name;return n.length>40?n.substring(0,40)+"\u2026":n;}return"";},font:{size:9}},title:{display:true,text:"Event Label"},grid:{display:false},reverse:true}}},
    plugins:[{id:"hl",afterDraw(chart){const{ctx,scales:{x,y},data}=chart,ds=data.datasets[0];ctx.save();ctx.font="9px sans-serif";ctx.textAlign="center";ctx.textBaseline="middle";ds.data.forEach(p=>{const xp=x.getPixelForValue(p.x),yp=y.getPixelForValue(p.y),bw=x.getPixelForValue(p.x+.5)-x.getPixelForValue(p.x-.5),bh=Math.abs(y.getPixelForValue(p.y+.5)-y.getPixelForValue(p.y-.5));if(p.v>0&&bw>20&&bh>15){ctx.fillStyle=getTextColor(p.v);if(p.v>=HIGH_PROBABILITY_THRESHOLD)ctx.font="bold 11px sans-serif";else if(p.v>=MEDIUM_PROBABILITY_THRESHOLD)ctx.font="bold 10px sans-serif";ctx.fillText((p.v*100).toFixed(0)+"%",xp,yp);if(p.v>=HIGH_PROBABILITY_THRESHOLD){ctx.strokeStyle="#FFD700";ctx.lineWidth=2;ctx.strokeRect(xp-bw/2,yp-bh/2,bw,bh);}}else if(p.v===0&&bw>15&&bh>10){ctx.fillStyle="#CCC";ctx.font="8px sans-serif";ctx.fillText("\u2014",xp,yp);}});ctx.restore();}}]
  });
  container.appendChild(canvas);
  const legend=document.createElement("div");legend.className="heatmap-legend";legend.innerHTML='<span>0%</span><div class="heatmap-gradient"></div><span>100%</span>';container.appendChild(legend);
  return container;
}
'@

Ensure-Fragment "fragments\js\timeline_chart.js" "Timeline chart" @'
// ===== Events Timeline =====
function createEventsTimeline(chunks, topEventNames) {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  const times = chunks.map(c=>((c.start_time||0)+(c.end_time||0))/2);
  const datasets = [];
  topEventNames.forEach((event,idx)=>{
    const color = EVENT_COLORS[idx % EVENT_COLORS.length];
    const probs = chunks.map(c=>{const p=(c.top_predictions||[]).find(p=>p.name===event.name);return p?p.prob:null;});
    datasets.push({label:event.name.length>50?event.name.substring(0,50)+"\u2026":event.name,data:probs.map((p,i)=>({x:times[i],y:p})),borderColor:color,backgroundColor:color,borderWidth:2,pointRadius:probs.map(p=>p!==null?getMarkerSize(p):0),pointBackgroundColor:color,pointBorderColor:"#FFF",pointBorderWidth:1,tension:.2,spanGaps:false,fill:false});
    const hpp=[];probs.forEach((p,i)=>{if(p!==null&&p>=HIGH_PROBABILITY_THRESHOLD)hpp.push({x:times[i],y:p});});
    if(hpp.length>0)datasets.push({label:null,data:hpp,backgroundColor:color+"33",borderColor:"transparent",pointRadius:hpp.map(()=>getMarkerSize(HIGH_PROBABILITY_THRESHOLD)*2),pointBackgroundColor:color+"33",pointBorderWidth:0,showLine:false,order:1});
  });
  const vt=times.filter(t=>t!=null),mn=vt.length>0?Math.min(...vt):0,mx=vt.length>0?Math.max(...vt):1;
  datasets.push({label:`Threshold (${(DEFAULT_PROBABILITY_THRESHOLD*100).toFixed(0)}%)`,data:[{x:mn,y:DEFAULT_PROBABILITY_THRESHOLD},{x:mx,y:DEFAULT_PROBABILITY_THRESHOLD}],borderColor:"#FF9800",borderDash:[5,5],borderWidth:1.5,pointRadius:0,fill:false,order:0});
  datasets.push({label:`High (${(HIGH_PROBABILITY_THRESHOLD*100).toFixed(0)}%)`,data:[{x:mn,y:HIGH_PROBABILITY_THRESHOLD},{x:mx,y:HIGH_PROBABILITY_THRESHOLD}],borderColor:"#F44336",borderDash:[2,4],borderWidth:1,pointRadius:0,fill:false,order:0});
  charts.timeline = new Chart(ctx,{type:"scatter",data:{datasets},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{position:"right",labels:{font:{size:9},filter:i=>i.text!==null}},tooltip:{callbacks:{label:ctx=>{if(ctx.dataset.label?.includes("Threshold")||ctx.dataset.label?.includes("High"))return null;if(ctx.raw.y===null)return"No data";return`${ctx.dataset.label}: ${(ctx.raw.y*100).toFixed(1)}%`;}}}},scales:{y:{min:-.05,max:1.05,ticks:{callback:v=>`${(v*100).toFixed(0)}%`},title:{display:true,text:"Probability"}},x:{title:{display:true,text:"Time (seconds)"}}}}});
  return canvas;
}
'@

Ensure-Fragment "fragments\js\results_bar.js" "Results bar chart" @'
// ===== Results Bar =====
function createResultsBar(topEventNames) {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  const rev = [...topEventNames].reverse();
  const labels = rev.map(e=>e.name.length>60?e.name.substring(0,60)+"\u2026":e.name);
  const probs = rev.map(e=>e.avgProb);
  charts.resultsBar = new Chart(ctx,{type:"bar",data:{labels,datasets:[{label:"Mean Probability",data:probs,backgroundColor:probs.map(p=>getBarStyle(p).color),borderColor:probs.map(p=>getBarStyle(p).borderColor),borderWidth:probs.map(p=>getBarStyle(p).borderWidth),borderRadius:3}]},options:{indexAxis:"y",responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false},tooltip:{callbacks:{label:ctx=>{const e=rev[ctx.dataIndex];return[`Avg: ${(ctx.parsed.x*100).toFixed(1)}%`,`Occurrences: ${e.count} chunks`,`Max: ${(e.maxProb*100).toFixed(1)}%`];}}}},scales:{x:{min:0,max:1.15,ticks:{callback:v=>`${(v*100).toFixed(0)}%`},title:{display:true,text:"Mean Probability"},grid:{color:"#F0F0F0"}},y:{ticks:{font:{size:10}}}}},plugins:[{id:"tl",afterDraw(chart){const{ctx,scales:{x,y}}=chart,tx=x.getPixelForValue(DEFAULT_PROBABILITY_THRESHOLD);ctx.save();ctx.strokeStyle="#FF9800";ctx.lineWidth=1.5;ctx.setLineDash([5,5]);ctx.beginPath();ctx.moveTo(tx,y.top);ctx.lineTo(tx,y.bottom);ctx.stroke();ctx.fillStyle="#FF9800";ctx.font="8px sans-serif";ctx.fillText(`Threshold (${(DEFAULT_PROBABILITY_THRESHOLD*100).toFixed(0)}%)`,tx+5,y.top+12);ctx.restore();}},{id:"bl",afterDraw(chart){const{ctx,scales:{x},data}=chart,meta=chart.getDatasetMeta(0);ctx.save();meta.data.forEach((bar,i)=>{const p=data.datasets[0].data[i];let l,f,c;if(p>=HIGH_PROBABILITY_THRESHOLD){l=`\u2605 ${(p*100).toFixed(0)}%`;f="bold 12px sans-serif";c="#1A1A6E";}else if(p>=MEDIUM_PROBABILITY_THRESHOLD){l=`${(p*100).toFixed(0)}%`;f="bold 11px sans-serif";c="#333";}else if(p>=DEFAULT_PROBABILITY_THRESHOLD){l=`${(p*100).toFixed(0)}%`;f="10px sans-serif";c="#555";}else{l=`${(p*100).toFixed(0)}%`;f="9px sans-serif";c="#888";}ctx.font=f;ctx.fillStyle=c;ctx.textBaseline="middle";ctx.fillText(l,bar.x+8,bar.y);ctx.font="8px sans-serif";ctx.fillStyle=p>=HIGH_PROBABILITY_THRESHOLD?"#F44336":"#999";ctx.textAlign="right";ctx.fillText(`#${data.labels.length-i}`,bar.x-35,bar.y);});ctx.restore();}}]});
  return canvas;
}
'@

Ensure-Fragment "fragments\js\chunks_summary.js" "Chunks summary grid" @'
// ===== Chunks Summary Grid =====
function createChunksSummary(chunks, topN) {
  const container = document.createElement("div");
  container.className = "plot-container xlarge";
  container.style.cssText = "display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:10px;overflow-y:auto;";
  const MAX = 24;
  chunks.slice(0,MAX).forEach((chunk,idx)=>{
    const card = document.createElement("div");
    card.style.cssText = "border:1px solid #DDD;border-radius:5px;padding:8px;background:#FAFAFA;position:relative;min-height:150px;";
    const preds = (chunk.top_predictions||[]).slice(0,topN);
    const maxP = preds.length>0?Math.max(...preds.map(p=>p.prob)):0;
    if(maxP>=HIGH_PROBABILITY_THRESHOLD){card.style.borderColor="#FFD700";card.style.borderWidth="2px";card.style.boxShadow="0 0 6px rgba(255,215,0,.3)";}
    else if(maxP>=MEDIUM_PROBABILITY_THRESHOLD){card.style.borderColor="#648FFF";card.style.borderWidth="1.5px";}
    else if(maxP>=DEFAULT_PROBABILITY_THRESHOLD)card.style.borderColor="#888";
    else card.style.borderColor="#DDD";
    const title = document.createElement("div");
    title.style.cssText = "font-weight:bold;font-size:10px;margin-bottom:5px;color:#333;";
    title.textContent = `C${chunk.chunk_index||idx+1}: ${(chunk.start_time||0).toFixed(1)}-${(chunk.end_time||0).toFixed(1)}s`;
    card.appendChild(title);
    if(preds.length===0){const nd=document.createElement("div");nd.textContent="No predictions";nd.style.cssText="color:#999;font-size:10px;text-align:center;padding:20px;";card.appendChild(nd);}
    else {
      const mc=document.createElement("canvas");mc.style.cssText="width:100%;height:120px;";card.appendChild(mc);
      const rp=[...preds].reverse();
      new Chart(mc.getContext("2d"),{type:"bar",data:{labels:rp.map(p=>p.name.length>25?p.name.substring(0,25)+"\u2026":p.name),datasets:[{data:rp.map(p=>p.prob),backgroundColor:rp.map(p=>getBarStyle(p.prob).color),borderColor:rp.map(p=>getBarStyle(p.prob).borderColor),borderWidth:rp.map(p=>getBarStyle(p.prob).borderWidth),borderRadius:2}]},options:{indexAxis:"y",responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false},tooltip:{callbacks:{label:ctx=>`${(ctx.parsed.x*100).toFixed(1)}%`}}},scales:{x:{min:0,max:1.1,ticks:{callback:v=>v===0?"0%":v===1?"100%":"",font:{size:7}}},y:{ticks:{font:{size:7},autoSkip:false}}}});
      const tp=rp[rp.length-1].prob;
      const lbl=document.createElement("div");lbl.style.cssText="position:absolute;top:30px;right:5px;font-size:9px;";
      lbl.style.fontWeight=tp>=HIGH_PROBABILITY_THRESHOLD?"bold":"normal";
      lbl.style.color=tp>=HIGH_PROBABILITY_THRESHOLD?"#1A1A6E":"#555";
      lbl.textContent=(tp>=HIGH_PROBABILITY_THRESHOLD?"\u2605 ":"")+(tp*100).toFixed(0)+"%";
      card.appendChild(lbl);
    }
    container.appendChild(card);
  });
  if(chunks.length>MAX){const n=document.createElement("div");n.style.cssText="grid-column:1/-1;text-align:center;color:#999;font-size:12px;padding:10px;";n.textContent=`Showing first ${MAX} of ${chunks.length} chunks`;container.appendChild(n);}
  return container;
}
'@

Ensure-Fragment "fragments\js\detail_card.js" "Segment detail card" @'
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
'@

Ensure-Fragment "fragments\js\main_app.js" "Tags page main render and init" @'
// ===== Tags Page: Main Render & Init =====
async function refreshPlots() {
  const grid = document.getElementById("plotsGrid");
  grid.innerHTML = '<div class="loading">Loading visualizations</div>';
  destroyCharts();
  try {
    const data = await fetchAndProcessData();
    const chunks = data.chunks || [], segments = data.segments || [], stats = data.stats || {};
    const isFiltered = data.isFiltered || false;
    const topN = parseInt(document.getElementById("topNSelect")?.value || "5");
    updateSummaryCards(stats, chunks, segments, isFiltered);
    if (isFiltered) {
      if (chunks.length===0) { grid.innerHTML='<div class="plot-card full-width"><div class="error-message"><h3>🔍 Segment Not Found</h3><p>The segment was not found.</p><a href="/tags" class="btn btn-small" style="margin-top:10px;">📊 View All</a></div></div>'; return; }
      grid.innerHTML=""; grid.appendChild(createSegmentDetailCard(chunks[0]));
      const te = getTopEventNames(chunks, topN);
      if (te.length>0) { const p=document.createElement("div");p.className="plot-card full-width";p.innerHTML='<h2>🔝 Top Predictions</h2><div class="plot-container"></div>';p.querySelector(".plot-container").appendChild(createResultsBar(te));grid.appendChild(p); }
      return;
    }
    if (chunks.length===0) { grid.innerHTML='<div class="plot-card full-width"><div class="info-message"><h3>📭 No Data</h3><p>No chunks found. Try adjusting filters.</p></div></div>'; return; }
    grid.innerHTML="";
    const topEvents = getTopEventNames(chunks, topN);
    if (topEvents.length===0) { grid.innerHTML='<div class="plot-card full-width"><div class="info-message"><h3>📊 No Predictions</h3><p>Chunks exist but have no prediction data.</p></div></div>'; return; }
    const p1=document.createElement("div");p1.className="plot-card full-width";p1.innerHTML=`<h2>🔥 Event Probability Heatmap</h2><p style="color:#666;font-size:.85em;margin-bottom:10px;">Segments: ${segments.length} | Chunks: ${chunks.length} | Top Events: ${topN}</p>`;p1.appendChild(createChunkHeatmap(chunks, topEvents));grid.appendChild(p1);
    const p2=document.createElement("div");p2.className="plot-card full-width";p2.innerHTML='<h2>📈 Event Probabilities Over Time</h2><p style="color:#666;font-size:.85em;margin-bottom:10px;">Marker size ∝ probability</p><div class="plot-container large"></div>';p2.querySelector(".plot-container").appendChild(createEventsTimeline(chunks, topEvents));grid.appendChild(p2);
    const p3=document.createElement("div");p3.className="plot-card";p3.innerHTML='<h2>📊 Aggregated Results</h2><p style="color:#666;font-size:.85em;margin-bottom:10px;">★ = High confidence</p><div class="plot-container"></div>';p3.querySelector(".plot-container").appendChild(createResultsBar(topEvents));grid.appendChild(p3);
    const p4=document.createElement("div");p4.className="plot-card full-width";p4.innerHTML=`<h2>📋 Per-Chunk Top-${Math.min(topN,3)} Predictions</h2><p style="color:#666;font-size:.85em;margin-bottom:10px;">Border color = confidence</p>`;p4.appendChild(createChunksSummary(chunks, Math.min(topN,3)));grid.appendChild(p4);
  } catch(e) { console.error(e); grid.innerHTML=`<div class="plot-card full-width"><div class="error-message"><h3>❌ Error</h3><p>${escapeHtml(e.message)}</p><button class="btn btn-small" onclick="refreshPlots()" style="margin-top:10px;">🔄 Retry</button></div></div>`; }
}
document.addEventListener("DOMContentLoaded",()=>{applySegmentFilterFromUrl();updateSegmentInfo();refreshPlots();setInterval(()=>{if(!activeSegmentFilter)refreshPlots();},60000);});
window.addEventListener("resize",()=>{Object.values(charts).forEach(c=>{try{c.resize();}catch(e){}});});
'@

# -- DASHBOARD-SPECIFIC JS FRAGMENT --
Ensure-Fragment "fragments\js\dashboard_main.js" "Dashboard page main logic" @'
// ===== Dashboard Page: Main Logic =====

// DOM elements
const uploadArea = document.getElementById("uploadArea");
const fileInput = document.getElementById("fileInput");
const tagBtn = document.getElementById("tagBtn");
const speechBtn = document.getElementById("speechBtn");
const resultsDiv = document.getElementById("results");
const progressDiv = document.getElementById("progress");
const progressFill = document.getElementById("progressFill");
const progressText = document.getElementById("progressText");
const toast = document.getElementById("toast");

// ===== File Upload Handling =====
uploadArea.addEventListener("click", () => fileInput.click());
uploadArea.addEventListener("dragover", (e) => { e.preventDefault(); uploadArea.classList.add("dragover"); });
uploadArea.addEventListener("dragleave", () => { uploadArea.classList.remove("dragover"); });
uploadArea.addEventListener("drop", (e) => {
  e.preventDefault(); uploadArea.classList.remove("dragover");
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});
fileInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) handleFile(file);
});

function handleFile(file) {
  selectedFile = file;
  uploadArea.querySelector("h3").textContent = `📁 ${file.name}`;
  uploadArea.querySelector("p").textContent = `Size: ${(file.size/1024).toFixed(1)} KB | Type: ${file.type}`;
  tagBtn.disabled = false;
  speechBtn.disabled = false;
}

// ===== Toast notification =====
function showToast(message, duration = 2000) {
  toast.textContent = message;
  toast.classList.add("show");
  setTimeout(() => { toast.classList.remove("show"); }, duration);
}

// ===== Navigate to segment analytics =====
function navigateToSegment(segmentDir, segmentNumber) {
  const segmentId = segmentDir || `seg_${String(segmentNumber || 0).padStart(3, "0")}`;
  showToast(`🔍 Opening analytics for ${segmentId} in new tab...`);
  window.open(`/tags?segment=${encodeURIComponent(segmentId)}`, "_blank");
}

// ===== Load Saved Chunks =====
async function loadChunks() {
  const chunksLoading = document.getElementById("chunksLoading");
  const filterSelect = document.getElementById("filterSelect");
  const limitSelect = document.getElementById("limitSelect");
  chunksLoading.style.display = "block";

  const speechOnly = filterSelect.value === "speech";
  const limit = parseInt(limitSelect.value);
  try {
    const url = `/tags/chunks?limit=${limit}&offset=0&speech_only=${speechOnly}`;
    const response = await fetch(url);
    const data = await response.json();
    updateStatistics(data.stats);
    updateChunksTable(data.chunks, data.total_entries, data.returned_entries);
    updateTopPredictions(data.stats.top_predictions);
  } catch (error) {
    document.getElementById("chunksTableContainer").innerHTML =
      `<div style="padding:20px;text-align:center;color:#721c24;">❌ Error loading data: ${escapeHtml(error.message)}</div>`;
  } finally {
    chunksLoading.style.display = "none";
  }
}

function updateStatistics(stats) {
  if (!stats) return;
  document.getElementById("totalSegments").textContent = stats.total_segments || 0;
  document.getElementById("speechSegments").textContent = stats.speech_segments || 0;
  document.getElementById("speechPercentage").textContent = (stats.speech_percentage || 0) + "%";
  document.getElementById("avgSpeechProb").textContent = (stats.avg_speech_probability || 0).toFixed(3);
}

function updateChunksTable(chunks, totalEntries, returnedEntries) {
  const container = document.getElementById("chunksTableContainer");
  if (!chunks || chunks.length === 0) {
    container.innerHTML = `<div style="padding:40px;text-align:center;color:#666;"><p style="font-size:1.2em;">📭 No saved segments found</p><p style="margin-top:10px;">Process some audio segments to see data here.</p></div>`;
    return;
  }
  let html = `<p style="margin-bottom:10px;color:#666;">Showing ${chunks.length} of ${totalEntries} segments <span style="font-size:0.85em;margin-left:10px;">💡 Click a row to view detailed analytics</span></p>
    <table><thead><tr><th>Segment</th><th>#</th><th>Speech</th><th>Probability</th><th>Mode</th><th>Top Predictions</th><th>Timestamp</th></tr></thead><tbody>`;
  chunks.forEach(chunk => {
    const speechDetected = chunk.speech_detected;
    const speechProb = chunk.speech_probability || 0;
    const predictions = chunk.top_predictions || [];
    const predText = predictions.map(p => `${p.name} (${(p.prob*100).toFixed(0)}%)`).join(", ");
    const timestamp = chunk.timestamp ? new Date(chunk.timestamp).toLocaleString() : "N/A";
    const segmentDir = chunk.segment_dir || "N/A";
    const segmentNumber = chunk.segment_number || "-";
    html += `<tr onclick="navigateToSegment('${escapeHtml(String(chunk.segment_dir||""))}', ${chunk.segment_number||0})" title="Click to view detailed analytics" style="cursor:pointer;">
      <td><strong>${escapeHtml(segmentDir)}</strong><span class="view-indicator">🔍</span></td>
      <td>${segmentNumber}</td>
      <td><span class="badge ${speechDetected?"badge-success":"badge-danger"}">${speechDetected?"✅ Yes":"❌ No"}</span></td>
      <td><div class="prob-bar"><div class="prob-fill ${speechDetected?"speech":""}" style="width:${speechProb*100}%"></div></div><small>${(speechProb*100).toFixed(1)}%</small></td>
      <td><span class="badge badge-info">${escapeHtml(chunk.processing_mode||"unknown")}</span></td>
      <td><small>${escapeHtml(predText)||"No predictions"}</small></td>
      <td><small>${escapeHtml(timestamp)}</small></td></tr>`;
  });
  html += "</tbody></table>";
  container.innerHTML = html;
}

function updateTopPredictions(predictions) {
  const container = document.getElementById("topPredictionsChart");
  if (!predictions || predictions.length === 0) {
    container.innerHTML = '<p style="text-align:center;color:#666;">No predictions data available</p>';
    return;
  }
  const maxCount = Math.max(...predictions.map(p => p.count));
  let html = "";
  predictions.slice(0, 10).forEach(pred => {
    const pct = ((pred.count/maxCount)*100).toFixed(0);
    html += `<div class="chart-bar"><div class="chart-label">${escapeHtml(pred.name)}</div><div class="chart-track"><div class="chart-fill" style="width:${pct}%"></div></div><div class="chart-value">${pred.count} (avg ${(pred.avg_probability*100).toFixed(0)}%)</div></div>`;
  });
  container.innerHTML = html;
}

// ===== Fill config and speech classes from API =====
async function loadConfig() {
  try {
    const resp = await fetch("/tags/config");
    const config = await resp.json();
    const configTable = document.getElementById("configTable");
    configTable.innerHTML = `
      <tr><td>Top K Predictions</td><td><strong>${config.top_k}</strong></td></tr>
      <tr><td>Speech Threshold</td><td><strong>${config.speech_prob_threshold}</strong></td></tr>
      <tr><td>Chunk Duration</td><td><strong>${config.chunk_duration}s</strong></td></tr>
      <tr><td>Chunk Overlap</td><td><strong>${config.chunk_overlap}s</strong></td></tr>
      <tr><td>Min Chunk Duration</td><td><strong>${config.min_chunk_duration}s</strong></td></tr>`;
    const speechList = document.getElementById("speechClassesList");
    const classes = config.speech_classes || SPEECH_CLASS_NAMES;
    speechList.innerHTML = classes.map(c => `<li style="padding:5px 0;border-bottom:1px solid #f0f0f0">• ${c}</li>`).join("");
  } catch(e) { console.error("Failed to load config:", e); }
}

// ===== Tagging and Speech Check =====
tagBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  const formData = new FormData();
  formData.append("file", selectedFile);
  const isChunked = document.getElementById("chunkedMode").checked;
  const endpoint = isChunked ? "/tags/chunks" : "/tags/audio";
  if (isChunked) {
    formData.append("chunk_duration", document.getElementById("chunkDuration").value);
    formData.append("overlap_duration", document.getElementById("overlapDuration").value);
  }
  showProgress("Processing audio...");
  try {
    const response = await fetch(endpoint, { method: "POST", body: formData });
    const data = await response.json();
    displayResults(data);
    setTimeout(() => loadChunks(), 500);
  } catch (error) { showError(error.message); }
  finally { hideProgress(); }
});

speechBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  const formData = new FormData();
  formData.append("file", selectedFile);
  showProgress("Checking speech...");
  try {
    const response = await fetch("/tags/speech-check", { method: "POST", body: formData });
    const data = await response.json();
    displaySpeechResult(data);
  } catch (error) { showError(error.message); }
  finally { hideProgress(); }
});

function showProgress(text) { progressDiv.style.display="block"; progressText.textContent=text; progressFill.style.width="50%"; }
function hideProgress() { progressDiv.style.display="none"; progressFill.style.width="0%"; }

function displayResults(data) {
  resultsDiv.style.display="block";
  let html="<h2>📊 Results</h2>";
  if (data.mode==="chunked"||data.chunks) {
    html+=`<p><strong>Mode:</strong> Chunked | <strong>Chunks:</strong> ${data.total_chunks} | <strong>Duration:</strong> ${data.total_duration_seconds}s</p><h3>Overall Top Predictions:</h3>`;
    data.overall_top_predictions.forEach(pred=>{html+=`<div class="result-item"><strong>${escapeHtml(pred.name)}</strong><div class="prob-bar"><div class="prob-fill" style="width:${pred.prob*100}%"></div></div><span>${(pred.prob*100).toFixed(1)}%</span></div>`;});
    if(data.chunks){html+="<h3>Chunk Details:</h3><table><tr><th>Chunk</th><th>Time</th><th>Top Prediction</th><th>Speech</th></tr>";data.chunks.forEach(chunk=>{const tp=chunk.predictions[0]||{name:"N/A",prob:0};html+=`<tr><td>${chunk.chunk_index}</td><td>${chunk.start_time}s-${chunk.end_time}s</td><td>${escapeHtml(tp.name)}(${(tp.prob*100).toFixed(1)}%)</td><td>${chunk.has_speech?"✅":"❌"}</td></tr>`;});html+="</table>";}
  } else {
    html+="<h3>Top Predictions:</h3>";
    data.top_predictions.forEach(pred=>{html+=`<div class="result-item"><strong>${escapeHtml(pred.name)}</strong><div class="prob-bar"><div class="prob-fill" style="width:${pred.prob*100}%"></div></div><span>${(pred.prob*100).toFixed(1)}%</span></div>`;});
  }
  html+=`<p><strong>Speech Detected:</strong> ${data.speech_detected?"✅ Yes":"❌ No"} | <strong>Processing Time:</strong> ${data.processing_time_seconds}s | <strong>RTF:</strong> ${data.real_time_factor}</p>`;
  resultsDiv.innerHTML=html;
}

function displaySpeechResult(data) {
  resultsDiv.style.display="block";
  resultsDiv.innerHTML=`<h2>🎤 Speech Detection Result</h2><div class="result-item"><h3>${data.has_speech?"✅ Speech Detected!":"❌ No Speech Detected"}</h3><p><strong>Probability:</strong> ${(data.speech_probability*100).toFixed(1)}%</p><p><strong>Threshold:</strong> ${(data.threshold_used*100).toFixed(0)}%</p><p><strong>File:</strong> ${escapeHtml(data.filename)}</p><p><strong>Processing Time:</strong> ${data.processing_time_seconds}s</p></div>`;
}

function showError(message) {
  resultsDiv.style.display="block";
  resultsDiv.innerHTML=`<div style="background:#fee;border:1px solid #fcc;padding:20px;border-radius:5px;"><h3 style="color:#c00;">❌ Error</h3><p>${escapeHtml(message)}</p></div>`;
}

// ===== Initial Load =====
document.addEventListener("DOMContentLoaded", () => {
  loadConfig();
  loadChunks();
  setInterval(loadChunks, 30000);
});
'@

Write-Host ""
Write-Host "  All fragments checked/created." -ForegroundColor Green

# ======================================================================
# STEP 4: Build both HTML files
# ======================================================================
Write-Host ""
Write-Host "Step 4: Building HTML files..." -ForegroundColor Yellow

Build-HtmlFile $TagsLayoutFile $TagsOutputFile "tags.html"
Build-HtmlFile $DashboardLayoutFile $DashboardOutputFile "dashboard.html"

# ======================================================================
# Summary
# ======================================================================
$tagsSize = (Get-Item $TagsOutputFile).Length
$dashSize = (Get-Item $DashboardOutputFile).Length

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Build complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  tags.html:      $([math]::Round($tagsSize/1024, 1)) KB" -ForegroundColor White
Write-Host "  dashboard.html: $([math]::Round($dashSize/1024, 1)) KB" -ForegroundColor White
Write-Host "  Total:          $([math]::Round(($tagsSize+$dashSize)/1024, 1)) KB" -ForegroundColor White
Write-Host "========================================" -ForegroundColor Cyan