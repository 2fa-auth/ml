import torch 
from torch.nn import Linear, Module, Tanh, ReLU


class SinNet(Module):
  def __init__(self, inp_dim, out_dim, type_model):
    super().__init__()

    self.type_model = type_model

    if self.type_model == 1:
      self.fc1 = Linear(inp_dim, 4)
      self.fc2 = Linear(4, out_dim)
      self.tanh = Tanh()
    else:
      self.fc1 = Linear(inp_dim, 32)
      self.fc2 = Linear(32, 16)
      self.fc3 = Linear(16, out_dim)
      self.relu = ReLU()

  def forward(self, x):
    if self.type_model == 1:
      out = self.tanh(self.fc1(x))
      out = self.fc2(out)
      return out
    
    out = self.relu(self.fc1(x))
    out = self.relu(self.fc2(out))
    out = self.fc3(out)
    return out


x_train = torch.linspace(-torch.pi, torch.pi, 100).unsqueeze(-1)
y_train = torch.sin(x_train).unsqueeze(-1)


model1 = SinNet(1, 1, type_model=1)
model2 = SinNet(1, 1, type_model=2)


st = model1.state_dict()
torch.save(st, "model1_weights.tar")

criterion = torch.nn.MSELoss()
optimizer_mod1 = torch.optim.SGD(params=model1.parameters(), lr=0.01)
optimizer_mod2 = torch.optim.SGD(params=model2.parameters(), lr=0.01)

num_epochs = 2000


for _e in range(num_epochs):
  loss_mean_mod1 = 0
  loss_cnt_mod1 = 0
  loss_mean_mod2 = 0
  loss_cnt_mod2 = 0

  for x, y in zip(x_train, y_train):
    pred1 = model1(x)
    pred2 = model2(x)
    loss1 = criterion(pred1, y)
    loss2 = criterion(pred2, y)

    optimizer_mod1.zero_grad()
    loss1.backward()
    optimizer_mod1.step()

    optimizer_mod2.zero_grad()
    loss2.backward()
    optimizer_mod2.step()

    loss_cnt_mod1 += 1
    loss_mean_mod1 = 1/loss_cnt_mod1 * loss1.item() + (1 - 1/loss_cnt_mod1) * loss_mean_mod1
    loss_cnt_mod2 += 1
    loss_mean_mod2 = 1/loss_cnt_mod2 * loss2.item() + (1 - 1/loss_cnt_mod2) * loss_mean_mod2

  if _e % 500 == 0:
    print(f"model A [{_e}/{num_epochs}] | loss_mean = {loss_mean_mod1}")
    print(f"moedl B [{_e}/{num_epochs}] | loss_mean = {loss_mean_mod2}")


x_test = torch.linspace(-3, 3, 100).unsqueeze(-1)
y_test = torch.sin(x)

loss_mean_mod1 = 0
loss_cnt_mod1 = 0
loss_mean_mod2 = 0
loss_cnt_mod2 = 0

with torch.no_grad():
  for x, y in zip(x_test, y_test):
    pred1 = model1(x)
    pred2 = model2(x)
    loss_mean_mod1 += criterion(pred1, y).item()
    loss_cnt_mod1 += 1

    loss_mean_mod2 += criterion(pred2, y).item()
    loss_cnt_mod2 += 1

loss_mean_mod1 /= loss_cnt_mod1
loss_mean_mod2 /= loss_cnt_mod2

print("\nRESULTS")
print(f"model A (4 нейрона, Tanh): ошибка = {loss_mean_mod1}")
print(f"model B (32->16, ReLU): ошибка = {loss_mean_mod2}")
print(f"difference between model A and model B: {loss_mean_mod1-loss_mean_mod2}")






