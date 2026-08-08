import torch
from torch.nn import Sequential, Linear, MSELoss

x_train = torch.linspace(-5, 5, 100).unsqueeze(1)
y_true = 2*x_train+1 # цель; НС должна научиться предугадывать значение y_true по входному X

model = Sequential(
  Linear(1, 1)
)


criterion = MSELoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001)
max_iterators = 5000

model.train()

for i in range(max_iterators):
  y_pred = model(x_train)
  loss = criterion(y_pred, y_true)

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  if i % 1000 == 0:
    print(f"[{i}/{max_iterators}] итераций пройдено")

model.eval()

print("\n")
for name, param in model.named_parameters():
    print(f"{name}: {param.data}")

 
x_test = torch.linspace(-10, 3, 50).unsqueeze(1)
print(x_test)
y_true = 2*x_test+1

d_test = zip(x_test, y_true)
loss_mean = 0
loss_cnt = 0

with torch.no_grad():
  for x_test, y_true in d_test:
    pred = model(x_test)
    loss_mean += criterion(pred, y_true).item()

    loss_cnt += 1
  loss_mean /= loss_cnt

print(loss_mean)



'''
1. подбирает рандомный вес
2. вычисляет pred
3. вычисляет среднюю ошибку между pred, y
4. находит градиент по средней ошибки
5. делает шаг в противоположную сторону от градиента (обновляет вес)
'''












