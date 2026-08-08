#!/usr/bin/env python3
import torch
import torch.nn.functional as F
import torchvision as tv
import torchvision.datasets
import torchvision.transforms.v2 as tfs

class Net(torch.nn.Module):
  def __init__(self):
    super().__init__()
  
    self.fc1 = torch.nn.Linear(784, 32)
    self.fc2 = torch.nn.Linear(32, 10)
    

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x

model = Net()

transforms = tfs.Compose([
  tfs.ToImage(),
  tfs.Grayscale(),
  tfs.ToDtype(dtype=torch.float32, scale=True),
  tfs.Lambda(lambda _x: _x.ravel())
])

# train_data = torch.utils.data.DataLoader(d_train, batch_size=32, shuffle=True)
dataset = torchvision.datasets.ImageFolder("dataset/train", transform=transforms)
d_train, d_val = torch.utils.data.random_split(dataset, [0.7, 0.3])
train_data = torch.utils.data.DataLoader(d_train, batch_size=32, shuffle=True)
val_data = torch.utils.data.DataLoader(d_val, batch_size=32, shuffle=False)



opt = torch.optim.Adam(params=model.parameters(), lr=0.01)
loss_foo = torch.nn.CrossEntropyLoss()
epochs = 20


for _e in range(epochs):
  model.train()
  loss_mean = 0
  lm_count = 0

  for x_train, y_train in train_data:
    pred = model(x_train)
    loss = loss_foo(pred, y_train)

    opt.zero_grad()
    loss.backward()
    opt.step()

    lm_count += 1
    loss_mean = 1/lm_count * loss.item() + (1 - 1/lm_count) * loss_mean
  model.eval()
  Q_val = 0
  count_val = 0

  for x_val, y_val in val_data:
    with torch.no_grad():
      p = model(x_val)
      loss = loss_foo(p, y_val)
      Q_val += loss.item()
      count_val += 1
  Q_val /= count_val

  print(f" | loss_mean={loss_mean:.3f}, Q_val={Q_val:.3f}")

d_test = torchvision.datasets.ImageFolder("dataset/test" , transform=transforms)
test_data = torch.utils.data.DataLoader(d_test, batch_size=500, shuffle=False)
Q = 0

model.eval()

for x_test, y_test in test_data:
  x = model(x_test)
  index_pred_tensor = torch.argmax(x, dim=1)
  Q += torch.sum(index_pred_tensor == y_test).item()

Q /= len(d_test)
print(Q)

'''
обучающая выборка разбивается на два независимых множества; на собественно обучающую выборку
и проверочную (то есть выборку валидации). То есть НС по прежнему обучается по обучающей выборки
но дополнительно после каждой эпохи вычисляется критерий качества работы нейронной сети 
по валидационной выборки.

Более коротко:
Валидация — нужна ТЕБЕ, чтобы настраивать модель во время обучения.
Тест — нужен ДЛЯ ОЦЕНКИ модели один раз в конце.
'''