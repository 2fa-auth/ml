#!/home/client/Documents/fun/py/PyTorch/lessons/prog3

import torch
import torchvision.models as models


dev = "cuda" if torch.cuda.is_available() else "cpu"

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(dev)

                             


