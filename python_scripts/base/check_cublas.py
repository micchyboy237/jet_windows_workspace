import torch
a = torch.randn(1000, 1000, device="cuda")
b = torch.randn(1000, 1000, device="cuda")
c = torch.matmul(a, b)        # uses cuBLAS under the hood
print("cuBLAS matmul successful →", c.device)