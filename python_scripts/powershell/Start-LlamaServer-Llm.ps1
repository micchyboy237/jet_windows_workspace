# Start-LlamaServer-Llm.ps1
# Interactive script to start llama-server with selected LLM model

Write-Host "Select an LLM model to start:`n"
Write-Host "1. Hermes-3-Llama-3.1-8B-Q4_K_M"
Write-Host "2. Qwen2.5-VL-7B-Instruct-Q4_K_M"
Write-Host "3. Qwen_Qwen3-4B-Instruct-2507-Q4_K_M"
Write-Host "4. DeepSeek-R1-Distill-Qwen-7B-Q4_K_M"
Write-Host "5. DeepSeek-Coder-V2-Lite-Instruct-IQ4_XS"
Write-Host ""

$modelChoice = Read-Host "Enter the number of your choice (1-5)"

switch ($modelChoice) {
    "1" {
        $modelName = "bartowski_Hermes-3-Llama-3.1-8B-GGUF_Hermes-3-Llama-3.1-8B-Q4_K_M.gguf"
        $command = "llama-server -m `"C:\Users\druiv\.cache\llama.cpp\$modelName`" --jinja --port 8080"
    }
    "2" {
        $modelName = "ggml-org_Qwen2.5-VL-7B-Instruct-GGUF_Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
        $mmproj = "ggml-org_Qwen2.5-VL-7B-Instruct-GGUF_mmproj-Qwen2.5-VL-7B-Instruct-Q8_0.gguf"
        $command = "llama-server -m `"C:\Users\druiv\.cache\llama.cpp\$modelName`" --mmproj `"C:\Users\druiv\.cache\llama.cpp\$mmproj`" --port 8080 --host 0.0.0.0 --n-gpu-layers 20"
    }
    "3" {
        $modelName = "bartowski_Qwen_Qwen3-4B-Instruct-2507-GGUF_Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
        $command = "llama-server -m `"C:\Users\druiv\.cache\llama.cpp\$modelName`" --jinja --port 8080 --host 0.0.0.0"
    }
    "4" {
        $modelName = "bartowski_DeepSeek-R1-Distill-Qwen-7B-GGUF_DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf"
        $command = "llama-server -m `"C:\Users\druiv\.cache\llama.cpp\$modelName`" --jinja --port 8080 --host 0.0.0.0"
    }
    "5" {
        $modelName = "bartowski_DeepSeek-Coder-V2-Lite-Instruct-GGUF_DeepSeek-Coder-V2-Lite-Instruct-IQ4_XS.gguf"
        $command = "llama-server -m `"C:\Users\druiv\.cache\llama.cpp\$modelName`" --jinja --port 8080 --host 0.0.0.0 --n-gpu-layers 10"
    }
    default {
        Write-Host "Invalid selection. Exiting..."
        exit 1
    }
}

$modelPath = "C:\Users\druiv\.cache\llama.cpp\$modelName"

if (-Not (Test-Path $modelPath)) {
    Write-Host "❌ Model file not found: $modelPath"
    exit 1
}

Write-Host "`n✅ Starting llama-server with model: $modelName"
Write-Host "---------------------------------------------`n"

# Display and run command
Write-Host "Running: $command`n"

# Start the server
Invoke-Expression $command
