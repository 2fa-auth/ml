import torch

x1 = torch.randn(3, 4, 5, 6)
chunks = x1.chunk(3)
print(chunks[0].shape) # размерность 1 4 5 6 

x2 = torch.cat(chunks, dim=0)
print(x1.equal(x2))