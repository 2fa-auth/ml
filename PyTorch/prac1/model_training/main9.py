import torch 
from torch.nn import Sequential, Linear, LeakyReLU

x_train = torch.linspace(-torch.pi, torch.pi, 100).unsqueeze(-1)
y_train = torch.sin(x_train).unsqueeze(-1)


model = Sequential(
  Linear(1, 16),
  LeakyReLU(),
  Linear(16, 1)
)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)

num_epochs = 2000

model.train()


for _e in range(num_epochs):
  loss_mean = 0
  loss_cnt = 0

  for x, y in zip(x_train, y_train):
    pred = model(x)
    loss = criterion(pred, y)

    optimizer.zero_grad()
    loss.backward()
    
    optimizer.step()

    loss_cnt += 1
    loss_mean = 1/loss_cnt * loss.item() + (1 - 1/loss_cnt) * loss_mean

  if _e % 500 == 0:
    print(f"[{_e}/{num_epochs}] | loss_mean = {loss_mean}")

model.eval()

x = torch.linspace(-3, 3, 100).unsqueeze(-1)
y = torch.sin(x).unsqueeze(-1)
data_test = zip(x, y)

loss_mean = 0
loss_cnt = 0

with torch.no_grad():
  for x, y in data_test:
    pred = model(x)
    loss_mean += criterion(pred, y).item() 
    loss_cnt += 1

loss_mean /= loss_cnt
print(loss_mean)




