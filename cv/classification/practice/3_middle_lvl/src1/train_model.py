#!/home/client/Documents/fun/py/venv/bin/python3
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
  
import torch
from torch.nn import Linear

# определение устройства  
dev = "cuda" if torch.cuda.is_available() else "cpu"


# инициализация модели 'resnet10'
weights_model = models.ResNet18_Weights.DEFAULT
transforms_model = weights_model.transforms()

model = models.resnet18(weights=weights_model)
model.requires_grad_(False) 

model.fc = Linear(512, 10)
model.fc.requires_grad_(True)

model = model.to(dev)

# трансформация выборки
dset_train = datasets.CIFAR10(root='.', train=True, transform=transforms_model, download=True)
d_train = DataLoader(dset_train, batch_size=32, shuffle=True)
# подготовка к обучению
optimizer = torch.optim.Adam(params=model.fc.parameters(), lr=0.001, weight_decay=0.001)
critetion = torch.nn.CrossEntropyLoss()
num_ep = 5
model.train()

# дообучение последнего (классифицированного) слоя  
for _e in range(num_ep):
  loss_mean = 0
  loss_cnt = 0
  for img, label in d_train:
    img, label = img.to(dev), label.to(dev)
    pred = model(img)
    loss = critetion(pred, label)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_cnt +=1
  loss_mean = 1/loss_cnt * loss.item() + (1 - 1/loss_cnt) * loss_mean
  print(f"[{_e}/{num_ep}] | loss_mean = {loss_mean:.5f}")

# сохранение весов 
torch.save(model.state_dict(), "model_transfer_resnet10")
model.eval()

# deb
print('\nмодель обучилась. веса сохранены')   
print('трансформация применимая к датасету CIFAR10 изображений:')
print(transforms_model)
