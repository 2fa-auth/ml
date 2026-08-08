import torch.nn as nn

model = nn.Sequential(
  nn.Linear(28*28, 32),
  nn.ReLU(),
  nn.Linear(32, 10)
) # <- при объявлении последовательной модели мы должны передавать объекты
  #    которые унаследованы от класса torch.nn.Module
print(model)

# 2 способ; чтобы можно было с помощью переменных обращаться к отдельным 
# слоям этой послед. модели:

model = nn.Sequential()
model.add_module("fc1", nn.Linear(28*28, 32))
model.add_module("relu", nn.ReLU())
model.add_module("fc2", nn.Linear(32, 10))

print(model)

# где model как составная часть более сложной модели:
class DigitNN(nn.Module):
  def __init__(self, inp, num, out):
    super().__init__()

    self.net = nn.Sequential(
      nn.Linear(inp, num),
      nn.ReLU(),
      nn.Linear(num, out)
    )

  def forward(self, x):
    return self.net(x)
  

# пример где нужна необходимость класса nn.ModuleList 
class ModelNN(nn.Module):
  def __init__(self, inp, out, nfcs=3):
    super().__init__()

    self.fcs = [nn.Linear(inp // n, inp // (n+1)) for n in range(1, nfcs+1)] # проблема с регестрацией! (pyTorch попросту не заглядывает во внутренность списка self.fcs и поэтому веса этих слоев обновляться не будут!)
    sz_inp = self.fcs[-1].out_features
    self.fc_out = nn.Linear(sz_inp, out) # <- зарегестрирован этот слой (добавлен во внутренний список _modules в классе nn.Module)

  def forward(self, x):
    for fc in self.fcs:
      x = fc(x)
      x = nn.functional.tanh(x)
    x = self.fc_out(x)
    return x
  
model = ModelNN(28*28, 10)
print(f"without reg {model}")

# пример использования nn.ModuleList
class ModelNN(nn.Module):
  def __init__(self, inp, out, nfcs=3):
    super().__init__()

    self.fcs = nn.ModuleList([nn.Linear(inp // n, inp // (n+1)) for n in range(1, nfcs+1)])
    sz_inp = self.fcs[-1].out_features
    self.fc_out = nn.Linear(sz_inp, out) 

  def forward(self, x):
    for fc in self.fcs:
      x = fc(x)
      x = nn.functional.tanh(x)
    x = self.fc_out(x)
    return x
  
model = ModelNN(28*28, 10)
print(f"with reg {model}")

# класс nn.ModuleList + add_module() (для имени)

class ModelNN(nn.Module):
  def __init__(self, inp, out, nfcs=3):
    super().__init__()
    self.fcs = nn.ModuleList()
    for n in range(1, nfcs+1):
      self.fcs.add_module(f"fc{n}", nn.Linear(inp // n, inp // (n+1)))

    self.fc_out = nn.Linear(inp // (nfcs+1), out) 

  def forward(self, x):
    for fc in self.fcs:
      x = fc(x)
      x = nn.functional.tanh(x)
    x = self.fc_out(x)
    return x
  
model = ModelNN(28*28, 10)
print(f"len list model.parameters: {len(list(model.parameters()))}")
print(f"fc1 (of fcs): {model.fcs.fc1}")
print(f"fc3 (of fcs): {model.fcs.fc3}")
# print(f"fc4 (of fcs): {model.fcs.fc4}") Error. fc4 is not exist  


# класс nn.ModuleDict (регестрирует отдельные слои но хранит их в виде словаря):
class ModelNN(nn.Module):
  def __init__(self, inp, out, nfcs=3, act_type=None):
    super().__init__()
    self.fcs = nn.ModuleList()
    self.act_type = act_type # тип функции активации
    for n in range(1, nfcs+1):
      self.fcs.add_module(f"fc{n}", nn.Linear(inp // n, inp // (n+1)))

    self.fc_out = nn.Linear(inp // (nfcs+1), out) 
    self.act_lst = nn.ModuleDict({
      "relu": nn.ReLU(),
      "lk_relu": nn.LeakyReLU(),
    })

  def forward(self, x):
    for fc in self.fcs:
      x = fc(x)
      # x = nn.functional.tanh(x)
      if self.act_type and self.act_type in self.act_lst:
        x = self.act_lst[self.act_type](x)

    x = self.fc_out(x)
    return x
  
model = ModelNN(28*28, 10, act_type="relu")
print(model)