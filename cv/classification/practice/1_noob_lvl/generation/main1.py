import torch 

X0 = torch.randn(500, 2)
X1 = torch.randn(500, 2) + 1

X = torch.cat([X0, X1], dim=0) # (1000, 2)

y = torch.cat([torch.zeros(500), torch.ones(500)], dim=0).unsqueeze(-1) # (1000, 1)

indices = torch.randperm(1000)
X = X[indices]
y = y[indices]

rand_indices = torch.randperm(100)[:1000]
y[rand_indices] = 1 - y[rand_indices]




