#!/home/client/Documents/fun/py/venv/bin/python3
import torchvision.models as models
import torch


model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

print(model)