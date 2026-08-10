import torch 

print(torch.randn(4, 1, 6, 1, 3).squeeze().unsqueeze(-1).shape)