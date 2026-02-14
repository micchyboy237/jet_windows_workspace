# Start-LlamaServer-Embedding.ps1
# Optimized for: Ryzen 5 3600 + GTX 1660 (6GB) + 16GB RAM

Write-Host "Starting llama.cpp embedding server launcher (optimized for short-text workloads)" -ForegroundColor Cyan

Write-Host "Select an embedding model to start:`n" -ForegroundColor Cyan
Write-Host "1. embeddinggemma-300M-Q8_0"
Write-Host "2. nomic-embed-text-v1.5.Q4_K_M          (recommended for balance)"
Write-Host "3. nomic-embed-text-v2-moe.Q4_K_M"
Write-Host "4. all-MiniLM-L12-v2-q4_0               (fastest, lower quality)"
Write-Host ""

$modelChoice = Read-Host "Enter the number of your choice (1-4)"

$flashAttnDefault = "on"
$batchDefault     = 512
$ubatchDefault    = 512

switch ($modelChoice) {
    "1" {
        $modelName = "embeddinggemma-300M-Q8_0.gguf"
        $b         = $batchDefault
        $ub        = $ubatchDefault
        $gpu       = -1
        $vramEst   = "~140-180 MB"
    }
    "2" {
        $modelName = "nomic-embed-text-v1.5.Q4_K_M.gguf"
        $b         = $batchDefault
        $ub        = $ubatchDefault
        $gpu       = -1
        $vramEst   = "~170-220 MB"
    }
    "3" {
        $modelName = "nomic-embed-text-v2-moe.Q4_K_M.gguf"
        $b         = $batchDefault
        $ub        = $ubatchDefault
        $gpu       = -1
        $vramEst   = "~300-380 MB"
    }
    "4" {
        $modelName = "all-MiniLM-L12-v2-q4_0.gguf"
        $b         = $batchDefault
        $ub        = $ubatchDefault
        $gpu       = -1
        $vramEst   = "~120-160 MB"
    }
    default {
        Write-Host "Invalid selection. Please choose 1-4." -ForegroundColor Red
        exit 1
    }
}

$useFlashAttn = Read-Host "`nUse flash attention? [Y/n] (recommended: Y for better memory efficiency)"
$flash = if ($useFlashAttn -match '^[nN]') { "off" } else { "on" }

$modelPath = "C:\Users\druiv\.cache\llama.cpp\embed_models\$modelName"

if (-not (Test-Path $modelPath)) {
    Write-Host "Model file not found: $modelPath" -ForegroundColor Red
    exit 1
}

Write-Host "`nSummary:" -ForegroundColor Cyan
Write-Host "  Model          : $modelName"
Write-Host "  Batch / ubatch : $b / $ub"
Write-Host "  Flash Attention: $flash"
Write-Host "  GPU layers     : $gpu (full offload)"
Write-Host "  Est. VRAM      : $vramEst"
Write-Host ""

$confirm = Read-Host "Start server with these settings? [Y/n]"
if ($confirm -match '^[nN]') {
    Write-Host "Aborted." -ForegroundColor Yellow
    exit 0
}

Write-Host "`nStarting embedding server with model: $modelName" -ForegroundColor Green
Write-Host "Context: $b tokens | GPU: Full Offload | Flash-attn: $flash" -ForegroundColor Yellow
Write-Host "---------------------------------------------`n"

# --- Optimized Command ---
$cmd = "llama-server.exe " +
       "-m `"$modelPath`" " +
       "--embedding " +
       "--host 0.0.0.0 --port 8001 " +
       "-c $b -ub $ub -b $b " +
       "--n-gpu-layers $gpu " +
       "--threads 6 --threads-batch 6 " +
       "--mlock --no-mmap " +
       "--flash-attn $flash " +
       "--cache-type-k q8_0 --cache-type-v q8_0 " +
       "--cont-batching " +
       "--parallel 4 " +
       "--threads-http 4 " +
       "--log-file `"C:\Users\druiv\.cache\logs\llama.cpp\embedding_logs`" " +
       "--log-colors on " +
       "--log-timestamps " +
       "--log-prefix " +
       "--verbose "

# Optional: quieter logs (uncomment if desired)
# $cmd += " --log-disable"

Write-Host "Running: $cmd`n" -ForegroundColor Gray

# Start server
Invoke-Expression $cmd
