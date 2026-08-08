import torch
import torch.nn.functional as F
import torchvision.datasets
import torchvision.transforms.v2 as tfs

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(784, 32)
        self.fc2 = torch.nn.Linear(32, 10)
    

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = Net()

'''
st = model.state_dict() # получить словарь где ключи - имена слоев, а значения - тензоры с числами и смещениями
print(st)

torch.save(st, "model_dnn.tar") # сохранить словарь в файл
st_load = torch.load("model_dnn.tar", weights_only=True) # st_load (выгруженный словарь) = st (загруженный)
model.load_state_dict(st_load) # !! Важно! Модель должна быть той же архитектуры, что и при сохранении
# сохранять и выгружать из файла можно не только словари но и другие типы данных

# Пример;
s = "hello"

torch.save(s, "ex.tar")
s_load = torch.load("ex.tar", weights_only=True)
print(s_load) # "hello"

# чтобы явно указать тип устройства в котором хочу загрузить данные:
t = torch.load("ex.tar", weights_only=True, map_location="cpu")

'''

transforms = tfs.Compose([
    tfs.ToImage(),
    tfs.Grayscale(),
    tfs.ToDtype(dtype=torch.float32, scale=True),
    tfs.Lambda(lambda _x: _x.ravel())
])

d_train = torchvision.datasets.ImageFolder("dataset", transform=transforms)
train_data = torch.utils.data.DataLoader(d_train, batch_size=32, shuffle=True)

opt = torch.optim.Adam(params=model.parameters(), lr=0.01)
loss_foo = torch.nn.CrossEntropyLoss()

# epochs = 2

model_data = torch.load("model_dnn_0.tar", weights_only=True)
model.load_state_dict(model_data["model"])
transforms.load_state_dict(model_data['tfs'])
opt.load_state_dict(model_data['opt'])

# model.train()

model_state_dict = {
    "tfs": transforms.state_dict(),
    "opt": opt.state_dict(),
    "model": model.state_dict()
}

# best_loss = 1e10 # заведомо высокое значение

# for _e in range(epochs):
#     loss_mean = 0
#     lm_count = 0

#     for x_train, y_train in train_data:
#         pred = model(x_train)
#         loss = loss_foo(pred, y_train)

#         opt.zero_grad()
#         loss.backward()
#         opt.step()

#         lm_count += 1
#         loss_mean = 1/lm_count + loss.item() + (1 - 1/lm_count) * loss_mean

#     if best_loss > loss_mean * 1.1:
#         best_loss = loss_mean
#         model_state_dict["model"] = model.state_dict()
#         torch.save(model_state_dict, f'model_dnn_{_e}.tar')

d_test = torchvision.datasets.ImageFolder("dataset" , transform=transforms)
test_data = torch.utils.data.DataLoader(d_test, batch_size=500, shuffle=False)
Q = 0

# model.eval()

for x_test, y_test in test_data:
    x = model(x_test)
    index_pred_tensor = torch.argmax(x, dim=1)
    Q += torch.sum(index_pred_tensor == y_test).item()

Q /= len(d_test)
print(Q)

