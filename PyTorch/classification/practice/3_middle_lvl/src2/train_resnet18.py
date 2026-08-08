#!/home/client/Documents/fun/py/venv/bin/python3
import torchvision.datasets
import torchvision.models 

import torch 
from torch.nn import Linear, CrossEntropyLoss 
from torch.utils.data import DataLoader

# определение процессора (центральный / графический) и дебаг 
dev = "cuda" if torch.cuda.is_available() else "cpu"
print('используемое устройство:', dev)

# определение модели (resnet18)
model_weights = torchvision.models.ResNet18_Weights.DEFAULT
model_transforms = model_weights.transforms()
model = torchvision.models.resnet18(weights=model_weights)

for param in model.parameters():
  param.requires_grad = False

model.fc = Linear(512, 2)

model.fc.requires_grad_(True)
model = model.to(dev)
 
# определение выборки
set_train = torchvision.datasets.ImageFolder('dataset/train', transform=model_transforms)
data_train = DataLoader(set_train, batch_size=32, shuffle=True)
set_test = torchvision.datasets.ImageFolder('dataset/test', transform=model_transforms)
data_test = DataLoader(set_test, batch_size=16, shuffle=False)

# обучение
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001, weight_decay=0.001)
critetion = torch.nn.CrossEntropyLoss()
num_ep = 5

model.train()
print(f'\nпошел процесс обучения. количество эпох {num_ep}, размер обучающей выборки {len(set_train)}, размер тестовой выборки {len(set_test)}')
for _e in range(num_ep):
  loss_mean = 0
  loss_cnt = 0
  for img, label in data_train:
    img, label = img.to(dev), label.to(dev)
    predicate = model(img)
    loss = critetion(predicate, label)
    # оптимизация
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_cnt += 1
  loss_mean = 1/loss_cnt * loss.item() + (1 - 1/loss_cnt) * loss_mean
  print(f"[{_e}/{num_ep}] | loss_mean = {loss_mean:.5f}")


like_file = 'model_cats_vs_dogs.pth'
torch.save(model.state_dict(), like_file)
print('\nмодель обучилась, веса сохранены в файл ')

# тестирование 
model.eval()

total_loss=0
correct=0
total=0

for img, label in data_test:
  img, label = img.to(dev), label.to(dev)
  predicate = model(img)

  total_loss += critetion(predicate, label).item()

  _, indexpred = torch.max(predicate, 1)
  total += label.size(0)
  correct += (label == indexpred).sum().item()

avg_loss = total_loss / len(data_test)
accuracy = correct / total
print(f'средняя потеря: {avg_loss:.5f}')
print(f"Правильных ответов: {correct} из {total}")
print(f"Точность: {accuracy:.4f} ({accuracy*100:.2f}%)")





