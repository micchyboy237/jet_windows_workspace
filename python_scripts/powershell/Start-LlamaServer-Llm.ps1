# Start-LlamaServer-Llm.ps1
# Optimized for: Ryzen 5 3600 + GTX 1660 (6GB) + 16GB RAM

Write-Host "Select an LLM model to start:`n" -ForegroundColor Cyan
Write-Host "1. Hermes-3-Llama-3.1-8B-Q4_K_M"
Write-Host "2. Qwen2.5-VL-7B-Instruct-Q4_K_M"
Write-Host "3. Qwen_Qwen3-4B-Instruct-2507-Q4_K_M"
Write-Host "4. DeepSeek-R1-Distill-Qwen-7B-Q4_K_M"
Write-Host "5. DeepSeek-Coder-V2-Lite-Instruct-IQ4_XS"
Write-Host "6. Llama-3.2-3B-Instruct-Q4_K_M"
Write-Host "7. Mistral-Nemo-Instruct-2407-Q4_K_M"
Write-Host "8. gemma-3-4b-it-Q4_K_M"
Write-Host ""

$modelChoice = Read-Host "Enter the number of your choice (1-8)"

switch ($modelChoice) {
    "1" { $modelName = "bartowski_Hermes-3-Llama-3.1-8B-GGUF_Hermes-3-Llama-3.1-8B-Q4_K_M.gguf"; $ub = 8192;  $gpu = 35; $jinja = $true; $mmproj = $null }
    "2" { $modelName = "ggml-org_Qwen2.5-VL-7B-Instruct-GGUF_Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"; $ub = 4096;  $gpu = 30; $jinja = $false; $mmproj = "ggml-org_Qwen2.5-VL-7B-Instruct-GGUF_mmproj-Qwen2.5-VL-7B-Instruct-Q8_0.gguf" }
    "3" { $modelName = "bartowski_Qwen_Qwen3-4B-Instruct-2507-GGUF_Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf"; $ub = 32768; $gpu = 40; $jinja = $true; $mmproj = $null }
    "4" { $modelName = "bartowski_DeepSeek-R1-Distill-Qwen-7B-GGUF_DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf"; $ub = 8192;  $gpu = 35; $jinja = $true; $mmproj = $null }
    "5" { $modelName = "bartowski_DeepSeek-Coder-V2-Lite-Instruct-GGUF_DeepSeek-Coder-V2-Lite-Instruct-IQ4_XS.gguf"; $ub = 16384; $gpu = 40; $jinja = $true; $mmproj = $null }
    "6" { $modelName = "bartowski_Llama-3.2-3B-Instruct-GGUF_Llama-3.2-3B-Instruct-Q4_K_M.gguf"; $ub = 8192;  $gpu = 999; $jinja = $true; $mmproj = $null }
    "7" { $modelName = "bartowski_Mistral-Nemo-Instruct-2407-GGUF_Mistral-Nemo-Instruct-2407-Q4_K_M.gguf"; $ub = 8192;  $gpu = 35; $jinja = $true; $mmproj = $null }
    "8" { $modelName = "ggml-org_gemma-3-4b-it-GGUF_gemma-3-4b-it-Q4_K_M.gguf"; $ub = 8192;  $gpu = 30; $jinja = $false; $mmproj = "ggml-org_gemma-3-4b-it-GGUF_mmproj-model-f16.gguf" }
    default { Write-Host "Invalid choice!" -ForegroundColor Red; exit 1 }
}

$modelPath = "C:\Users\druiv\.cache\llama.cpp\$modelName"
$mmprojPath = if ($mmproj) { "C:\Users\druiv\.cache\llama.cpp\$mmproj" } else { $null }

if (-Not (Test-Path $modelPath)) { Write-Host "Model not found: $modelPath" -ForegroundColor Red; exit 1 }

# --- Optimized Command ---
$cmd = "llama-server.exe " +
       "-m `"$modelPath`" " +
       "--host 0.0.0.0 --port 8080 " +
       "--ctx-size 4096 " +
    #    "-c $ub -ub $ub -b 512 " +           # Context, ubatch, batch
    #    "--n-gpu-layers $gpu " +             # GPU offload
    #    "--threads 6 --threads-batch 6 " +   # Match CPU cores
    #    "--mlock --no-mmap " +               # RAM lock, avoid pagefile
       "--flash-attn on " +                 # Critical for speed
       "--cache-type-k q8_0 --cache-type-v q8_0 "  # Q8 KV cache = faster + accurate
    #    "--cont-batching"                    # Dynamic batching

if ($jinja) { $cmd += " --jinja" }
if ($mmprojPath -and (Test-Path $mmprojPath)) { $cmd += " --mmproj `"$mmprojPath`"" }

Write-Host "`nModel: $modelName" -ForegroundColor Green
Write-Host "Context: $ub | GPU Layers: $gpu | KV Cache: Q8_0" -ForegroundColor Yellow
Write-Host "Running: $cmd`n" -ForegroundColor Gray

Invoke-Expression $cmd
