import torch 

m = torch.randn(2, 3)
print(m)

m = m.unsqueeze(0).unsqueeze(2).unsqueeze(-1)
m = m.repeat(5, 1, 4, 1, 6)
print(m.shape)
print(m)