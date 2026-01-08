# Optional clean (safe)
pip uninstall -y torch torchvision torchaudio lightning pytorch-lightning torchcodec torchmetrics

# Install CUDA Torch explicitly
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install pyannote.audio (uses default PyPI, keeps your CUDA Torch)
pip install pyannote.audio --no-deps
