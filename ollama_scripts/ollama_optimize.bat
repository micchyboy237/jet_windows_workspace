ollama pull qwen3:4b-q4_K_M
ollama pull granite3.3:2b
ollama pull gemma3:4b-it-q4_K_M
ollama pull qwen2.5vl:3b-q4_K_M
ollama pull deepseek-coder-v2:16b-lite-instruct-q3_K_M
ollama pull phi-4-mini-reasoning:3.8b-q4_K_M
ollama pull gemma3n:e2b-it-q4_K_M
ollama pull mistral:7b-instruct-v0.3-q2_K
ollama pull qwen3:1.7b-q4_K_M


Requires CPU offloading (much slower)
ollama pull mistral-nemo:12b-instruct-2407-q2_K
ollama pull mistral-small3.2:24b-instruct-2506-q4_K_M


Optional
ollama pull qwen3:4b-instruct-2507-q4_K_M


# llama.cpp
llama-server -hf ggml-org/gemma-3-4b-it-GGUF
llama-server -hf bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF:Q4_K_M
llama-server -hf ggml-org/Qwen2.5-VL-7B-Instruct-GGUF:Q4_K_M
llama-server --jinja -fa -hf bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M
llama-server --jinja -fa -hf bartowski/Mistral-Nemo-Instruct-2407-GGUF:Q4_K_M
llama-server --jinja -fa -hf bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF:Q4_K_M \
    --chat-template-file models/templates/llama-cpp-deepseek-r1.jinja
llama-server --jinja -fa -hf bartowski/Hermes-3-Llama-3.1-8B-GGUF:Q4_K_M \
    --chat-template-file models/templates/NousResearch-Hermes-3-Llama-3.1-8B-tool_use.jinja
