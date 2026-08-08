#!/home/client/Documents/fun/py/venv/bin/python3
from torchvision.datasets import CIFAR10
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

import torch
from torch.nn import Linear, CrossEntropyLoss
from torch.utils.data import DataLoader


dev = "cuda" if torch.cuda.is_available() else "cpu" 

weights_model = EfficientNet_B0_Weights.DEFAULT
transforms_model = weights_model.transforms()

model = efficientnet_b0(weights=weights_model)
for param in model.parameters():
  param.requires_grad = False

model.classifier = Linear(1280, 100)
model.classifier.requires_grad_(True)
model = model.to(dev)
print(model.classifier) #архитектура классифицированного слоя модели

#выборка
train_set = CIFAR10(root='.', train=True, transform=transforms_model, download=True)
test_set = CIFAR10(root='.', train=False, transform=transforms_model, download=True)
train_data = DataLoader(train_set, batch_size=32, shuffle=True)
test_data = DataLoader(test_set, batch_size=8, shuffle=False)

optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001, weight_decay=0.001)
criterion = CrossEntropyLoss()


#обучение
model.train()
numep=5

for _e in range(numep):
  loss_cnt=0
  loss_mean=0
  for img, lab in train_data:
    img, lab = img.to(dev), lab.to(dev)
    predicate = model(img)
    loss = criterion(predicate, lab)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_cnt +=1
  loss_mean = 1/loss_cnt * loss.item() + (1 - 1/loss_cnt) * loss_mean
  print(f'[{_e+1}/{numep}] loss mean {loss_mean}')

#тест
model.eval()
loss_mean=0
total=0
correct=0
with torch.no_grad():
  for img, lab in test_data:
    img, lab = img.to(dev), lab.to(dev)
    predicate = model(img)
    loss_mean += criterion(predicate, lab).item()

    total += lab.size(0)
    _, index = torch.max(predicate, 1)
    correct += (index == lab).sum().item()


loss_mean /= len(test_data)

print(f'mean error: {loss_mean}')
print(f'correct responce {correct} from {total}')
