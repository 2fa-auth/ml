#!/home/client/Documents/fun/py/venv/bin/python3
import torchvision.models as models
from torch.nn.functional import softmax
import torch 
import torchvision.transforms.v2 as tfs_v2

from PIL import Image


dev = "cuda" if torch.cuda.is_available() else "cpu"


vgg_weights = models.VGG16_Weights.DEFAULT
cats = vgg_weights.meta['categories']

trans1 = vgg_weights.transforms() # трансформация класса DEFAULT
trans2 = tfs_v2.Compose([ #  трансоформация пользовательская
  tfs_v2.ToImage(),
  tfs_v2.Resize(256),
  tfs_v2.CenterCrop(224),
  tfs_v2.ToDtype(dtype=torch.float32, scale=True),
  tfs_v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

img = Image.open("images1/sport-car.jpg").convert("RGB")
img_net = trans2(img).unsqueeze(0).to(dev) # добавление батча (1): [1, 3, 224, 224]


model = models.vgg16(weights=models.VGG16_Weights.DEFAULT).to(dev)

model.eval()
p = model(img_net).squeeze() # (1000) вместо (1, 1000)
res = p.softmax(dim=0).sort(descending=True) # (value, index)

mccn = model.features # модель CNN
mp = model.classifier # клафициированная модель (перцептрон)

for s, i in zip(res[0][:5], res[1][:5]):
  print(f"{cats[i]}: {s:.4f}")


  
# удаление весов из RAM
del model
torch.cuda.empty_cache() if torch.cuda.is_available() else None


import gc
gc.collect()

print(f"Модель удалена: {model is None}")  # Будет True - значит удалена

# Или просто:
try:
    print(model)
except NameError:
    print("Модель успешно удалена из памяти")
