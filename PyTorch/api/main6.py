import torch
import random

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc1 = torch.nn.Linear(2, 5)
        self.fc2 = torch.nn.Linear(5, 1)

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = Net()
    
    x_train = torch.FloatTensor([(20, 21), (21, 22), (22, 23), (23, 24), (24, 25), (25, 26), (26, 27), (27,28)])
    y_train = torch.FloatTensor([22, 23, 24, 25, 26, 27, 28, 29]).unsqueeze(1)

    loss_foo = torch.nn.MSELoss()
    opt = torch.optim.Adam(params=model.parameters(), lr=0.1)
    total = len(x_train)

    model.train() 

    losses = []

    print("веса ДО обучения: ")
    print("fc1.weight:", model.fc1.weight.data)
    print("fc2.weight:", model.fc2.weight.data)

    for _ in range(50000):
        k = random.randint(0, total-1)
        y = model(x_train[k])
        loss = loss_foo(y, y_train[k])
        losses.append(loss.item())

        opt.zero_grad()
        loss.backward() # на основе loss составляется градиент (вычисляются производные)
        opt.step()      # на основе градиента оптимизируются веса с шагом lr 
    
    print("веса ПОСЛЕ ОБУЧЕНИЯ:")
    print("fc1.weight:", model.fc1.weight.data)
    print("fc2.weight:", model.fc2.weight.data)
    model.eval()
    
    for x, y in zip(x_train, y_train):
        with torch.no_grad():
            y = model(x)
            print(y)

    # тот батч которая сеть еще не видела!
    x = torch.FloatTensor([(28, 29)])
    y = model(x)

    print(y.data)

    print("losses")
    print(losses[:10])
    print(losses[-10:])
    print(min(losses))