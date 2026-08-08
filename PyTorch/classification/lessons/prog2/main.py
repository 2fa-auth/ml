import torch
import torch.utils.data as data
import torchvision.transforms.v2 as tfs
import torch.nn as nn
import torch.optim as optim

import os
import json
from PIL import Image

class SunDataset(data.Dataset):
  def __init__(self, path, train=True, transform=None):
    self.path = os.path.join(path, "train" if train else "test")
    self.transform = transform

    # парсинг .json
    with open(os.path.join(self.path, "format.json"), "r") as fp:
      self.format = json.load(fp)

    # извлечение нужной информации из .json
    self.length = len(self.format)
    self.files = tuple(self.format.keys())
    self.targets = tuple(self.format.values())

  def __getitem__(self, item):
    path_file = os.path.join(self.path, self.files[item])
    img = Image.open(path_file).convert('RGB')

    if self.transform:
      img = self.transform(img)

    return img, torch.tensor(self.targets[item], dtype=torch.float32)
  
  def __len__(self):
    return self.length
  

transforms = tfs.Compose([
  tfs.ToImage(),
  tfs.ToDtype(torch.float32, scale=True) # scale - это масштабирование (все пиксели переводятся от 0 до 1)
]) 
d_train = SunDataset("dataset_reg", transform=transforms)
train_data = data.DataLoader(d_train, batch_size=32, shuffle=True)

model = nn.Sequential(
  nn.Conv2d(3, 32, 3, padding='same'),
  nn.ReLU(),
  nn.MaxPool2d(2),
  nn.Conv2d(32, 8, 3, padding='same'),
  nn.ReLU(),
  nn.MaxPool2d(2),
  nn.Conv2d(8, 4, 3, padding='same'),
  nn.ReLU(),
  nn.MaxPool2d(2),
  nn.Flatten(),
  nn.Linear(4096, 128),
  nn.ReLU(),
  nn.Linear(128, 2)
)

criterion = nn.MSELoss()
optimizer = optim.Adam(params=model.parameters(), lr=0.01, weight_decay=0.001)

num_epochs = 5
model.train()

for _e in range(num_epochs):
  loss_mean = 0
  lm_count = 0

  for x_train, y_train in train_data:
    pred = model(x_train)
    loss = criterion(pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    lm_count += 1
    loss_mean = 1/lm_count * loss.item() + (1 - 1/lm_count) * loss_mean
  print(f"Epoch [{_e+1}/{num_epochs}], loss_mean={loss_mean:.3f}")


st = model.state_dict()
torch.save(st, "model_sun.tar")

d_test = SunDataset("dataset_reg", train=False, transform=transforms)
test_data = data.DataLoader(d_test, batch_size=50, shuffle=False)

loss_mean = 0
loss_count = 0
model.eval()

with torch.no_grad():
  for x_test, y_test in test_data:
    p = model(x_test)
    loss_mean += criterion(p, y_test).item()
    loss_count += 1

loss_mean /= loss_count
print(loss_mean)