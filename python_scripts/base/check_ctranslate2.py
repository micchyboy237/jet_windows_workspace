# check_ctranslate2.py
import ctranslate2
import torch

print("=== PyTorch CUDA check ===")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version (torch): {torch.version.cuda}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print("\n=== CTranslate2 CUDA check ===")
print(f"Detected CUDA devices: {ctranslate2.get_cuda_device_count()}")  # This is the correct function

# Optional: force validation by creating a dummy model on GPU
try:
    # This will raise a clear exception if CUDA is misconfigured
    ctranslate2.Generator("does-not-exist", device="cuda", device_index=0)
    print("CTranslate2 can instantiate a model on CUDA → setup is OK")
except Exception as e:
    print(f"CTranslate2 CUDA test failed: {e}")