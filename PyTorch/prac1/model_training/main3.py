import torch

t = torch.arange(3*5*2*7).view(3, 5, 2, 7).permute(3, 1, 0, 2)
print(t.shape)