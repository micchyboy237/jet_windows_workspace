# llama-server --models-preset C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\models.embedders.ini --models-max 1 --host 0.0.0.0 --port 8081

# llama-server `
#   -m C:\Users\druiv\.cache\llama.cpp\embed_models\nomic-embed-text-v1.5.Q4_K_M.gguf `
#   --embedding `
#   --pooling mean `
#   -c 8192 `
#   -b 8192 `
#   -ub 2048 `
#   -np 8 `
#   -ngl 99 `
#   --rope-scaling yarn `
#   --rope-freq-scale 0.75 `
#   --host 0.0.0.0 `
#   --port 8081 `
#   --alias nomic-embed:1.5 `
#   --tags embed

llama-server `
  -m "C:\Users\druiv\.cache\llama.cpp\embed_models\nomic-embed-text-v1.5.Q4_K_M.gguf" `
  --embedding `
  --pooling mean `
  -c 8192 `
  -b 8192 `
  -ub 8192 `
  --rope-scaling yarn `
  --rope-freq-scale 0.75 `
  -ngl 99 `
  -np 4 `
  --host 0.0.0.0 `
  --port 8081 `
  --alias nomic-embed:1.5 `
  --tags embed
