#!/home/client/Documents/fun/py/venv/bin/python3
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torchvision.transforms.v2 as tfs
from torchvision.transforms import InterpolationMode
from torch.nn import Sequential, ReLU, Sigmoid, Conv2d, MaxPool2d, Flatten, Linear


class CSet(Dataset):
  def __init__(self, pathdir, train=True, transform=None):
    super().__init__()

    self.pathdir = os.path.join(pathdir, "train" if train else "test")
    self.transforms = transform
    class_images = [os.path.join(self.pathdir, i) for i in os.listdir(self.pathdir)]  
    self.train_data = {x: 1 if y%2==0 else 0 for y, x in enumerate(class_images) }

    self.length = len(self.train_data)
      
  def __getitem__(self, index):
    img = Image.open(list(self.train_data)[index])
    label = list(self.train_data.items())[index][1]

    if self.transforms:
      img = self.transforms(img)
    return img, label

    
  def __len__(self):
    return self.length    


transforms = tfs.Compose([
  tfs.ToImage(),
  tfs.Resize((128, 128), interpolation=InterpolationMode.BILINEAR),
  tfs.ToDtype(torch.float32, scale=True)
])


set_train = CSet("image", train=True, transform=transforms)
loader_train = DataLoader(set_train, batch_size=3, shuffle=True)

model = Sequential(
  Conv2d(3, 32, 3, stride=1, padding="same"), # 128, 128
  ReLU(),
  MaxPool2d(2), # 64 64 
  Conv2d(32, 16, 3, stride=1, padding="same"),   
  ReLU(),
  MaxPool2d(2), #  32 32
  Conv2d(16, 8, 3, stride=1, padding="same"), 
  ReLU(),
  MaxPool2d(2), # 16 16
  Flatten(),
  Linear(2048, 128),
  ReLU(),
  Linear(128, 1),
  Sigmoid()
)


opt = torch.optim.Adam(params=model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

model.train()
max_ep = 10
for _e in range(max_ep):
  for x_train, y_train in loader_train:

    y_train = y_train.float().view(-1, 1)
    pred = model(x_train)
    loss = criterion(pred, y_train)

    opt.zero_grad()
    loss.backward()
    opt.step()

  if _e % 2 == 0:
    print(f"[*] {_e}/{max_ep} ready!")


model.eval()

set_test = CSet("image", train=False, transform=transforms)
loader_test = DataLoader(set_test, batch_size=2, shuffle=False)
mae_metric = torch.nn.L1Loss()

loss_mean = 0
loss_cnt = 0
for x_test, y_test in loader_test:
  y_test = y_test.float().view(-1, 1)
  pred = model(x_test)
  loss_mean += mae_metric(pred, y_test).item()

  loss_cnt += 1

print(f"средняя ошибка: {loss_mean/loss_cnt}")





