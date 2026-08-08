#!/home/client/Documents/fun/py/venv/bin/python3
import torch
from torch.nn import Linear

import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
  


# определение устройства  
dev = "cuda" if torch.cuda.is_available() else "cpu"


# инициализация модели 'resnet10'
weights_model = models.ResNet18_Weights.DEFAULT
transforms_model = weights_model.transforms()
model = models.resnet18(weights=weights_model)
model.fc = Linear(512, 10)

# загрузка весов
model.load_state_dict(torch.load('model_transfer_resnet10.tar', weights_only=True))
model = model.to(dev)
model.eval()

dset_test = datasets.CIFAR10('.', train=False, transform=transforms_model, download=True)
d_test = DataLoader(dset_test, batch_size=16, shuffle=False)

# тестирование модели
total_loss = 0
correct = 0
total = 0

criterion = torch.nn.CrossEntropyLoss()

with torch.no_grad():
  for img, label in d_test:
    img, label = img.to(dev), label.to(dev)
    pred = model(img)

    total_loss += criterion(pred, label).item()

    _, predicted = torch.max(pred, 1)
    total += label.size(0)
    correct += (predicted == label).sum().item()

avg_loss = total_loss / len(d_test)
accuracy = correct / total  

print(f"Средняя потеря: {avg_loss:.5f}")
print(f"Правильных ответов: {correct} из {total}")
print(f"Точность: {accuracy:.4f} ({accuracy*100:.2f}%)")

