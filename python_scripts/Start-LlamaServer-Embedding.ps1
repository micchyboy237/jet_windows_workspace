# Start-LlamaServer-Embedding.ps1
# Interactive script to start llama-server with selected embedding model

Write-Host "Select an embedding model to start:`n"
Write-Host "1. embeddinggemma-300M-Q8_0"
Write-Host "2. nomic-embed-text-v1.5.Q4_K_M"
Write-Host "3. nomic-embed-text-v2-moe.Q4_K_M"
Write-Host ""

$modelChoice = Read-Host "Enter the number of your choice (1-3)"

switch ($modelChoice) {
    "1" {
        $modelName = "embeddinggemma-300M-Q8_0.gguf"
    }
    "2" {
        $modelName = "nomic-embed-text-v1.5.Q4_K_M.gguf"
    }
    "3" {
        $modelName = "nomic-embed-text-v2-moe.Q4_K_M.gguf"
    }
    default {
        Write-Host "Invalid selection. Exiting..."
        exit 1
    }
}

$modelPath = "C:\Users\druiv\.cache\llama.cpp\embed_models\$modelName"

if (-Not (Test-Path $modelPath)) {
    Write-Host "❌ Model file not found: $modelPath"
    exit 1
}

Write-Host "`n✅ Starting llama-server with model: $modelName"
Write-Host "---------------------------------------------`n"

# Build and run command
$cmd = "llama-server -m `"$modelPath`" --embedding --gpu-layers -1 --threads 6 -ub 2048 --batch-size 2048 --ctx-size 2048 --host 0.0.0.0 --port 8081"
Write-Host "Running: $cmd`n"

# Start the server
Invoke-Expression $cmd
