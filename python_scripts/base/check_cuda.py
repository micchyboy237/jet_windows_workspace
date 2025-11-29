import torch
print(torch.__version__)                  # e.g. 2.5.0+cu128 or 2.4.1+cu128
print(torch.cuda.is_available())          # → True
print(torch.version.cuda)                 # → 12.8
print(torch.cuda.get_device_name(0))      # → GeForce GTX 1660
print(torch.backends.cudnn.version())     # → 8902 or 9xxx