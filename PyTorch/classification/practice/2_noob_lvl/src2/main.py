#!/home/client/Documents/fun/py/venv/bin/python3
import torch 
import torchvision.datasets
import torchvision.transforms.v2 as transforms
import torch.utils.data as data
from torch.nn import Sequential, Linear, Conv2d, MaxPool2d, Flatten, ReLU, Dropout

_transforms = transforms.Compose([
  transforms.ToImage(),
  transforms.ToDtype(dtype=torch.float32, scale=True),
  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

cifar10set_train = torchvision.datasets.CIFAR10(root=".", train=True, download=True, transform=_transforms)
train_d = data.DataLoader(cifar10set_train, 64, shuffle=True)

model = Sequential(
  Conv2d(3, 32, 3, padding="same"),
  ReLU(),
  Conv2d(32, 32, 3, padding="same"),
  ReLU(),
  MaxPool2d(2),

  Conv2d(32, 64, 3, padding="same"),
  ReLU(),
  Conv2d(64, 64, 3, padding="same"),
  ReLU(),
  MaxPool2d(2),

  Conv2d(64, 128, 3, padding="same"),
  ReLU(),
  Conv2d(128, 128, 3, padding="same"),
  ReLU(),
  MaxPool2d(2),

  Flatten(),
  Linear(128*4*4, 256),
  ReLU(),
  Linear(256, 10)
)

optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

num_ep = 20


loss_mean = 0
lm_count = 0
for _e in range(num_ep):
  for xt, yt in train_d:
    pred = model(xt)
    loss = criterion(pred, yt)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    lm_count += 1
    loss_mean = 1/lm_count * loss.item() + (1 - 1/lm_count) * loss_mean

  print(f"[*] {_e}/{num_ep}, loss mean: {loss_mean}")
model.eval()


cifar10set_test = torchvision.datasets.CIFAR10(root=".", train=False, download=False, transform=_transforms)
test_d = data.DataLoader(cifar10set_test, 64, shuffle=True)

loss_mean = 0
loss_cnt = 0

with torch.no_grad():
  for xt, yt in test_d:
    pred = model(xt)
    loss_mean += criterion(pred, yt).item()
    loss_cnt += 1
loss_mean /= loss_cnt
print(f"Mean Absolute Error: {loss_mean}")


# mean loss = 0.3 (train), mean loss = 1.4 (test)




