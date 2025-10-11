# download_models.ps1
# Sequentially download all models so they’re cached before running llama-server

$models = @(
    "bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF:Q4_K_M",
    "ggml-org/Qwen2.5-VL-7B-Instruct-GGUF:Q4_K_M",
    "bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M",
    "bartowski/Mistral-Nemo-Instruct-2407-GGUF:Q4_K_M",
    "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF:Q4_K_M",
    "bartowski/Hermes-3-Llama-3.1-8B-GGUF:Q4_K_M"
)

foreach ($m in $models) {
    Write-Host "Downloading $m ..."
    llama-cli --hf-repo $m -p "ping" -n 1 --no-display-prompt
}
