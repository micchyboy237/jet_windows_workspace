# Step 1: Create output directory (if not exists)
New-Item -ItemType Directory -Force -Path "C:\Users\druiv\.cache\llama.cpp\embed_models"

# Step 2: Convert from HF cache to f16 GGUF
python "C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\llama-cpp-python\vendor\llama.cpp\convert_hf_to_gguf.py" `
  "C:\Users\druiv\.cache\huggingface\hub\models--sentence-transformers--all-MiniLM-L12-v2\snapshots\c004d8e3e901237d8fa7e9fff12774962e391ce5" `
  --outfile "C:\Users\druiv\.cache\llama.cpp\embed_models\all-MiniLM-L12-v2-f16.gguf" `
  --outtype f16

# Step 3: Quantize to Q4_0
python "C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\llama-cpp-python\examples\low_level_api\quantize.py" `
  "C:\Users\druiv\.cache\llama.cpp\embed_models\all-MiniLM-L12-v2-f16.gguf" `
  "C:\Users\druiv\.cache\llama.cpp\embed_models\all-MiniLM-L12-v2-q4_0.gguf" `
  2

llama-server -m "C:\Users\druiv\.cache\llama.cpp\embed_models\all-MiniLM-L12-v2-q4_0.gguf" --host 0.0.0.0 --port 8080 -ub 512 --embedding --pooling cls --n-gpu-layers 999
