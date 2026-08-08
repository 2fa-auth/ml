import torch


x = torch.tensor([2.0], requires_grad=True)
y = torch.tensor([-4.0], requires_grad=True)

f = (x + y) ** 2 + 2 * x * y # псевдо сеть
f.backward() # вычисляется градиент для каждого веса (если я изменю наприемр x или y то насколько изменится результат)

print(f)
print(x.data, x.grad)
print(y.data, y.grad)
