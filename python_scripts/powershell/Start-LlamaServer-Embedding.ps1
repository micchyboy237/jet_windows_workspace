# Start-LlamaServer-Embedding.ps1
# Optimized for: Ryzen 5 3600 + GTX 1660 (6GB) + 16GB RAM

Write-Host "Select an embedding model to start:`n" -ForegroundColor Cyan
Write-Host "1. embeddinggemma-300M-Q8_0"
Write-Host "2. nomic-embed-text-v1.5.Q4_K_M"
Write-Host "3. nomic-embed-text-v2-moe.Q4_K_M"
Write-Host "4. all-MiniLM-L12-v2-q4_0"
Write-Host ""

$modelChoice = Read-Host "Enter the number of your choice (1-4)"

switch ($modelChoice) {
    "1" {
        $modelName = "embeddinggemma-300M-Q8_0.gguf"
        $ub = 2048
        $pooling = "mean"
        $gpu = 999
    }
    "2" {
        $modelName = "nomic-embed-text-v1.5.Q4_K_M.gguf"
        $ub = 2048
        $pooling = "mean"
        $gpu = 999
    }
    "3" {
        $modelName = "nomic-embed-text-v2-moe.Q4_K_M.gguf"
        $ub = 2048
        $pooling = "mean"
        $gpu = 999
    }
    "4" {
        $modelName = "all-MiniLM-L12-v2-q4_0.gguf"
        $ub = 512
        $pooling = "cls"
        $gpu = 999
    }
    default {
        Write-Host "Invalid selection. Please choose 1-4." -ForegroundColor Red
        exit 1
    }
}

$modelPath = "C:\Users\druiv\.cache\llama.cpp\embed_models\$modelName"

if (-Not (Test-Path $modelPath)) {
    Write-Host "Model file not found: $modelPath" -ForegroundColor Red
    exit 1
}

Write-Host "`nStarting embedding server with model: $modelName" -ForegroundColor Green
Write-Host "Pooling: $pooling | Max tokens: $ub | GPU: Full Offload" -ForegroundColor Yellow
Write-Host "---------------------------------------------`n"

# --- Optimized Command ---
$cmd = "llama-server.exe " +
       "-m `"$modelPath`" " +
       "--embedding --pooling $pooling " +
       "--host 0.0.0.0 --port 8081 " +
       "-c $ub -ub $ub -b 512 " +                     # Context, ubatch, batch
       "--n-gpu-layers $gpu " +                       # Full offload
       "--threads 6 --threads-batch 6 " +             # Match CPU
       "--mlock --no-mmap " +                         # Prevent swapping
       "--flash-attn on " +                           # +30–50% speed
       "--cache-type-k q8_0 --cache-type-v q8_0 " +   # Best KV cache
       "--cont-batching"                              # Dynamic batching

Write-Host "Running: $cmd`n" -ForegroundColor Gray

# Start server
Invoke-Expression $cmd