import torch
import torchvision.transforms.v2 as tfs
import torch.optim as optim
import torch.nn.functional as F
from  torchvision.datasets import ImageFolder 
    

class DigNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(784, 32)
        self.fc2 = torch.nn.Linear(32, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


def main():
    model = DigNet()

    # to_tensor = tfs.ToImage() # просто перевести изображения в тензор недостаточно
    # d_train = DigitDataset("dataset", transform=to_tensor)

    transforms = tfs.Compose([
        tfs.ToImage(),
        tfs.Grayscale(),
        tfs.ToDtype(torch.float32, scale=True), # был диапозон от 0 до 255 и стал от 0 до 1
        tfs.Lambda(lambda _img: _img.ravel())
    ])

    d_train = ImageFolder("dataset", transform=transforms)

    train_data = torch.utils.data.DataLoader(d_train, batch_size=32, shuffle=True)
    
    opt = optim.Adam(params=model.parameters(), lr=0.01)
    loss_foo = torch.nn.CrossEntropyLoss()

    epochs = 2
    model.train()

    for _e in range(epochs):
        for x_train, y_train in train_data:
            pred = model(x_train)
            loss = loss_foo(pred, y_train)

            opt.zero_grad()
            loss.backward()
            opt.step()

    d_test = ImageFolder("dataset", transform=transforms)
    test_data = torch.utils.data.DataLoader(d_test, batch_size=500, shuffle=False)
    Q=0

    model.eval()

    for x_test, y_test in test_data:
        with torch.no_grad():
            p = model(x_test)
            p = torch.argmax(p, dim=1) # [500][10] = > [500] [index 1-й картинки, index 2-й картинки, .. ,index N-й картинки] 
            Q += torch.sum(p == y_test).item()

    Q /= len(d_test)
    print(Q)

    
if __name__ == "__main__":
    main()