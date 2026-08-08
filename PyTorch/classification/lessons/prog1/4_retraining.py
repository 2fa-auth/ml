'''
борьба с переобучением:
  L2, BN, Dropout (BN и Dropout нежелательно использовать на одном слое)
    Стоит использовать эти механизмы только если модель начинает переобучиваться 
    
'''

#!/usr/bin/python3
import torch 
import torchvision as tv
import torchvision.transforms.v2 as tfs
from torch.utils.data import DataLoader


class Net(torch.nn.Module):
  def __init__(self):
    super().__init__()

    self.fc1 = torch.nn.Linear(784, 128)
    self.bn = torch.nn.BatchNorm1d(128)
    # self.dropout = torch.nn.Dropout1d(0.3) # одномерный слой (вектор из 784 элементов) (но использовать только в самом крайнем случае)
    self.fc2 = torch.nn.Linear(128, 10)
    self.relu = torch.nn.ReLU()

  def forward(self, x):
    x = self.fc1(x)
    x = self.bn(x)
    x = self.relu(x)
    # out = self.dropout(out)

    x = self.fc2(x)

    return x
  
net = Net()

transforms = tv.transforms.Compose([
  tfs.ToImage(),
  tfs.Grayscale(),
  tfs.ToDtype(dtype=torch.float32, scale=True),
  tfs.Lambda(lambda _x: _x.ravel())
])

datasets_train = tv.datasets.ImageFolder("dataset/train", transform=transforms)
data_train, data_valid = torch.utils.data.random_split(datasets_train, [0.7, 0.3])
batches_train = DataLoader(data_train, batch_size=30, shuffle=True)
batches_valid = DataLoader(data_valid, batch_size=30, shuffle=False)


optim = torch.optim.Adam(params=net.parameters(), lr=0.01)
loss_foo = torch.nn.CrossEntropyLoss()
epochs = 10

for _e in range(epochs):
  train_loss_mean=0
  train_loss_cnt = 0

  net.train()
  for x, y in batches_train:
    pred = net(x)
    loss = loss_foo(pred, y)

    optim.zero_grad()
    loss.backward()
    optim.step()

    train_loss_cnt += 1
    train_loss_mean = 1/train_loss_cnt*loss.item()+(1-1/train_loss_cnt)*train_loss_mean
  
  net.eval()
  with torch.no_grad():
    valid_loss_mean = 0
    valid_loss_cnt = 0

    for x, y in batches_valid:
      pred = net(x)  
      loss = loss_foo(pred, y)
      
      valid_loss_mean += loss.item()
      valid_loss_cnt += 1

  valid_loss_mean /= valid_loss_cnt
  print(f"Era {_e+1} has ended. TRAIN Loss mean {train_loss_mean:.3f} VALID Loss mean {valid_loss_mean:.3f}")

datasets_test = tv.datasets.ImageFolder("dataset/test", transform=transforms)
batches_test = DataLoader(datasets_test, batch_size=500, shuffle=False)

Q = 0

for x, y in batches_test:
  with torch.no_grad():
    pred = net(x)
    index_pred = torch.argmax(pred, dim=1)
    Q += torch.sum(index_pred == y).item()

Q /= len(datasets_test)
print(Q)