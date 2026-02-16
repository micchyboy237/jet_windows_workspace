# Start-LlamaServer-Embedding.ps1
# Optimized for: Ryzen 5 3600 + GTX 1660 (6GB) + 16GB RAM - 2026 best practices
# 4 parallel workers, ctx scaled automatically per worker

Write-Host "Starting llama.cpp embedding server launcher (optimized for short-to-medium text workloads)" -ForegroundColor Cyan
Write-Host "Select an embedding model to start:`n" -ForegroundColor Cyan
Write-Host "1. embeddinggemma-300M-Q8_0         (fast, 2048 natural ctx)"
Write-Host "2. nomic-embed-text-v1.5.Q4_K_M     (long context capable)"
Write-Host "3. nomic-embed-text-v2-moe.Q4_K_M   (MOE variant)"
Write-Host "4. all-MiniLM-L12-v2-q4_0           (fastest, lower quality)"
Write-Host ""

$modelChoice = Read-Host "Enter the number of your choice (1-4)"

# ---- Runtime scaling variables ----
$parallel     = 4
$threadsHttp  = 4
$threads      = 10

switch ($modelChoice) {
    "1" {
        $modelName  = "embeddinggemma-300M-Q8_0.gguf"
        $naturalCtx = 2048
        $vramEst    = "~200-300 MB"
    }
    "2" {
        $modelName  = "nomic-embed-text-v1.5.Q4_K_M.gguf"
        $naturalCtx = 4096
        $vramEst    = "~250-400 MB"
    }
    "3" {
        $modelName  = "nomic-embed-text-v2-moe.Q4_K_M.gguf"
        $naturalCtx = 2048
        $vramEst    = "~350-480 MB"
    }
    "4" {
        $modelName  = "all-MiniLM-L12-v2-q4_0.gguf"
        $naturalCtx = 1024
        $vramEst    = "~150-220 MB"
    }
    default {
        Write-Host "Invalid selection. Please choose 1-4." -ForegroundColor Red
        exit 1
    }
}

# ---- Derived values (DRY + scalable) ----
$ctxSize   = $naturalCtx * $parallel
$batchSize = $ctxSize

$useFlashAttn = Read-Host "`nUse flash attention? [Y/n] (recommended: Y)"
$flash = if ($useFlashAttn -match '^[nN]') { "off" } else { "on" }

$modelPath = "C:\Users\druiv\.cache\llama.cpp\embed_models\$modelName"
if (-not (Test-Path $modelPath)) {
    Write-Host "Model file not found: $modelPath" -ForegroundColor Red
    exit 1
}

Write-Host "`nSummary:" -ForegroundColor Cyan
Write-Host " Model             : $modelName"
Write-Host " Natural Context   : $naturalCtx"
Write-Host " Parallel Workers  : $parallel"
Write-Host " Final Context     : $ctxSize"
Write-Host " Batch Size        : $batchSize"
Write-Host " Pooling           : cls"
Write-Host " Flash Attention   : $flash"
Write-Host " GPU layers        : -1 (full offload)"
Write-Host " Threads           : $threads"
Write-Host " HTTP Threads      : $threadsHttp"
Write-Host " KV cache quant    : f16 / f16"
Write-Host " Est. VRAM         : $vramEst"
Write-Host ""

$confirm = Read-Host "Start server with these settings? [Y/n]"
if ($confirm -match '^[nN]') {
    Write-Host "Aborted." -ForegroundColor Yellow
    exit 0
}

Write-Host "`nStarting embedding server with model: $modelName" -ForegroundColor Green
Write-Host "---------------------------------------------`n"

$cmd = "llama-server.exe " +
       "-m `"$modelPath`" " +
       "--embedding " +
       "--pooling cls " +
       "--host 0.0.0.0 --port 8001 " +
       "-c $ctxSize -ub $batchSize -b $batchSize " +
       "--n-gpu-layers -1 " +
       "--threads $threads --threads-batch $threads " +
       "--mlock --no-mmap " +
       "--flash-attn $flash " +
       "--cache-type-k f16 --cache-type-v f16 " +
       "--cont-batching " +
       "--parallel $parallel " +
       "--threads-http $threadsHttp " +
       "--metrics " +
       "--log-file `"C:\Users\druiv\.cache\logs\llama.cpp\embedding_logs`" " +
       "--log-colors on " +
       "--log-timestamps " +
       "--log-prefix " +
       "--verbose "

Write-Host "Running: $cmd`n" -ForegroundColor Gray

Invoke-Expression $cmd
