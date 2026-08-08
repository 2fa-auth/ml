import torch
from torch.nn import *

# train data
X = torch.randn(5000, 2) * 6-3
y = (torch.sin(X[:, 0]) * torch.cos(X[:, 1]) + 0.1 * torch.randn(5000)).unsqueeze(-1)

# building a model
model = Sequential(
  Linear(2, 64),
  LeakyReLU(),
  Linear(64, 128),
  LeakyReLU(),
  Linear(128, 64),
  LeakyReLU(),
  Linear(64, 1),
)

opt = torch.optim.Adam(params=model.parameters(), lr=0.01)
criterion = MSELoss()
mae_metric = L1Loss()

num_ep=10000

# train the model
model.train()
for _e in range(num_ep):
  pred = model(X)
  loss = criterion(pred, y)

  opt.zero_grad()
  loss.backward()
  opt.step()
  
  if _e % 500 == 0:
    with torch.no_grad():
      mae = mae_metric(pred, y).item()
    print(f"epochs [{_e}/{num_ep}] | MAE = {mae:.6f}")

model.eval()

# test data
Xt = torch.randn(1000, 2) * 6-3
yt = (torch.sin(Xt[:, 0]) * torch.cos(Xt[:, 1]) + 0.1 * torch.randn(1000)).unsqueeze(-1)

# testing the model
loss_sum = 0
loss_cnt = 0
with torch.no_grad():
  for x, y in zip(Xt, yt):
    pred = model(x)
    loss_sum += mae_metric(pred, y).item()
    loss_cnt += 1

print(f"mae {loss_sum/loss_cnt}")