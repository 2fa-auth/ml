import torch 
import torch.nn as nn


class0_x = torch.randn(200, 2) + torch.tensor([2., 2.])
class1_x = torch.randn(200, 2) + torch.tensor([-2., -2.])

x = torch.cat([class0_x, class1_x], dim=0)
y = torch.cat([torch.zeros(200, 1), torch.ones(200, 1)], dim=0)  

indices = torch.randperm(400)
x = x[indices]
y = y[indices]

model = nn.Sequential(
  nn.Linear(2, 16),
  nn.LeakyReLU(),
  nn.Linear(16, 16),
  nn.LeakyReLU(),
  nn.Linear(16, 1)
)

opt = torch.optim.Adam(params=model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()

num_epochs = 1000

# train
model.train()

for _e in range(num_epochs):
  loss_mean=0
  loss_cnt=0
  for x_train, y_train in zip(x, y):
    pred = model(x_train)
    loss = criterion(pred, y_train)

    opt.zero_grad()
    loss.backward()
    opt.step()

    loss_cnt += 1
    loss_mean = 1/loss_cnt * loss.item() + (1 - 1/loss_cnt) * loss_mean

  if _e % 200 == 0:
    print(f"[{_e}/{num_epochs}] | loss_mean = {loss_mean:.3f}")

model.eval()

# inference
test_class0 = torch.randn(50, 2) + torch.tensor([2., 2.])
test_class1 = torch.randn(50, 2) + torch.tensor([-2., -2.])

x = torch.cat([test_class0, test_class1], dim=0)
y = torch.cat([torch.zeros(50, 1), torch.ones(50,1)],dim=0)

indices_test = torch.randperm(100)
x = x[indices_test]
y = y[indices_test]

loss_mean = 0
loss_cnt = 0

print(y.flatten())

with torch.no_grad():
  for x_test, y_test in zip(x, y):
    pred = model(x_test)
    # print(pred, end=' ')
    if pred >= 1:
      print("1", end=' ')
    else:
      print("0", end=' ')
    loss_mean += criterion(pred, y_test).item()
    loss_cnt += 1
  
loss_mean /= loss_cnt
print(f"\nсредняя ошибка = {loss_mean}")