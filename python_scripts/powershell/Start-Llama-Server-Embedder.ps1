# llama-server --models-preset C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\models.embedders.ini --models-max 2 --host 0.0.0.0 --port 8081

# Basic embedding server with concurrency
llama-server `
  -m "C:\Users\druiv\.cache\llama.cpp\embed_models\nomic-embed-text-v2-moe.Q4_K_M.gguf" `
  --embeddings `
  --pooling mean `
  --parallel 4 `
  --batch-size 2048 `
  --ubatch-size 512 `
  --threads 8 `
  --ctx-size 512 `
  --host 0.0.0.0 `
  --port 8081 `
  --alias nomic-embed-2-moe `
  --tags embed

# llama-server `
#   -m "C:\Users\druiv\.cache\llama.cpp\embed_models\nomic-embed-text-v1.5.Q4_K_M.gguf" `
#   --embeddings `
#   --pooling mean `
#   --parallel 6 `
#   --batch-size 4096 `
#   --ubatch-size 512 `
#   --threads 6 `
#   --ctx-size 8192 `
#   --rope-scaling yarn `
#   --rope-freq-scale 0.75 `
#   --ngl 99 `
#   --host 0.0.0.0 `
#   --port 8080 `
#   --alias nomic-embed-1.5 `
#   --tags embed