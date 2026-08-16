import os
import json

from PIL import Image

import torch
import torchvision.transforms.v2 as tfs
import torch.optim as optim
import torch.nn.functional as F



class DigitDataset(torch.utils.data.Dataset):
    def __init__(self, path, train=True, transform=None):

        self.path = os.path.join(path, "train" if train else "test")
        self.transform = transform


        with open(os.path.join(self.path, "format.json"), "r") as fp:
            self.format = json.load(fp)
        
        self.length = 0
        self.files = []
        self.targets = torch.eye(10)

        for _dir, _target in self.format.items():
            path = os.path.join(self.path, _dir)
            list_files = os.listdir(path)
            self.length += len(list_files)
            for _x in list_files:
                full_path = os.path.join(path, _x)
                self.files.append((full_path, _target))

    def __getitem__(self, index):
        path_file, target = self.files[index]
        
        t = self.targets[target]
        img = Image.open(path_file)

        if self.transform:
            img = self.transform(img).ravel().float() / 255.0

        return img, t

    def __len__(self):
        return self.length
    

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

    to_tensor = tfs.ToImage()
    d_train = DigitDataset("dataset", transform=to_tensor)
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

    d_test = DigitDataset("dataset", train=False, transform=to_tensor)
    test_data = torch.utils.data.DataLoader(d_test, batch_size=500, shuffle=False)
    Q=0

    model.eval()

    for x_test, y_test in test_data:
        with torch.no_grad():
            p = model(x_test)
            p = torch.argmax(p, dim=1) # [500][10] = > [500] [index 1-й картинки, index 2-й картинки, .. ,index N-й картинки] 
            y = torch.argmax(y_test, dim=1)
            Q += torch.sum(p == y).item()

    Q /= len(d_test)
    print(Q)

    
if __name__ == "__main__":
    main()