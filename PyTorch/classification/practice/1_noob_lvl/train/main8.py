import torch
from torch.nn import Linear

X = torch.tensor([[0.4, 0.5], [0.9, 0.2], [0.2, 0.], [0.1, 0.]])
print(X)
print('\n\n')


model = Linear(2, 3)
print(model.weight)
y = model(X)
print(y)
print('\n')

model = Linear(3, 1)
print(model.weight)
y = model(y)
print('\n\nКОНЕЧНАЯ МАТРИЦА')
print(y)




"""

MAE -
L1Loss
MSELOSS

"""