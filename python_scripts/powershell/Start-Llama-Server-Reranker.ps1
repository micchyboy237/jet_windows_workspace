llama-server --models-preset C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\models.rerankers.ini --models-max 1 --host 0.0.0.0 --port 8082

# Basic reranking server with concurrency
# llama-server `
#   -m "C:\Users\druiv\.cache\llama.cpp\embed_models\bge-reranker-v2-m3.Q4_K_M.gguf" `
#   --rerank `
#   --pooling rank `
#   --parallel 4 `
#   --batch-size 512 `
#   --ubatch-size 128 `
#   --threads 6 `
#   --threads-batch 6 `
#   --ctx-size 512 `
#   --ngl 99 `
#   --host 0.0.0.0 `
#   --port 8082 `
#   --alias bge-reranker-v2-m3 `
#   --tags rerank
