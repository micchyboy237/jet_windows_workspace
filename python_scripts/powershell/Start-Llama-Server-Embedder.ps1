# llama-server --models-preset C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\models.embedders.ini --models-max 2 --host 0.0.0.0 --port 8081

# Basic embedding server with concurrency
llama-server `
  -m "C:\Users\druiv\.cache\llama.cpp\embed_models\nomic-embed-text-v2-moe.Q4_K_M.gguf" `
  --embeddings `
  --pooling mean `
  --parallel 4 `
  --ctx-size 2048 `
  --batch-size 512 `
  --ubatch-size 512 `
  --threads 8 `
  --host 0.0.0.0 `
  --port 8081 `
  --alias nomic-embed-2-moe `
  --tags embed

# llama-server `
#   -m "C:\Users\druiv\.cache\llama.cpp\embed_models\nomic-embed-text-v1.5.Q4_K_M.gguf" `
#   --embeddings `
#   --pooling mean `
#   --parallel 4 `
#   --ctx-size 4096 `
#   --batch-size 1024 `
#   --ubatch-size 1024 `
#   --threads 8 `
#   --host 0.0.0.0 `
#   --port 8082 `
#   --alias nomic-embed-v1.5 `
#   --tags embed

# For inputs longer than 2048 token, use rope scaling
# llama-server `
#   -m "C:\Users\druiv\.cache\llama.cpp\embed_models\nomic-embed-text-v1.5.Q4_K_M.gguf" `
#   --embeddings `
#   --pooling mean `
#   --parallel 2 `
#   --ctx-size 16384 `
#   --batch-size 8192 `
#   --ubatch-size 8192 `
#   --rope-scaling yarn `
#   --rope-freq-scale 0.75 `
#   --threads 8 `
#   --host 0.0.0.0 `
#   --port 8081 `
#   --alias nomic-embed-1.5 `
#   --tags embed