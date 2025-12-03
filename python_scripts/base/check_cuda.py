import torch
print(torch.__version__)                  # e.g. 2.5.0+cu128 or 2.4.1+cu128
print(torch.cuda.is_available())          # → True
print(torch.backends.cudnn.version())     # → 8902 or 9xxx
print(f"CUDA version: {torch.version.cuda}") # → 12.8
print(f"GPU count: {torch.cuda.device_count()}")
print(f"Current GPU: {torch.cuda.current_device()} ({torch.cuda.get_device_name(0)})") # → GeForce GTX 1660