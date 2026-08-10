import torch 

a = torch.randn(4, 3)
b = torch.randn(3,)
print(a + b)
print(a - a.mean(dim=1).unsqueeze(1))