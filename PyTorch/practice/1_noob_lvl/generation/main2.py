import torch

X = torch.randn(800, 2)
y = torch.where(X[:, 0] + X[:, 1] >= 0, 1, 0)
print(y)

class0_idx = torch.where(y == 0)[0]
class1_idx = torch.where(y == 1)[0]
print(class0_idx)

print(f"До балансировки: класс 0 = {len(class0_idx)}, класс 1 = {len(class1_idx)}")

if len(class0_idx) > 400:
    class0_idx = class0_idx[:400]

if len(class1_idx) > 400:
    class1_idx = class1_idx[:400]

if len(class0_idx) < 400:
    extra_idx = torch.randint(0, len(class0_idx), (400 - len(class0_idx),))
    class0_idx = torch.cat([class0_idx, class0_idx[extra_idx]])
if len(class1_idx) < 400:
    extra_idx = torch.randint(0, len(class1_idx), (400 - len(class1_idx),))
    class1_idx = torch.cat([class1_idx, class1_idx[extra_idx]])


balanced_idx = torch.cat([class0_idx, class1_idx])
X_balanced = X[balanced_idx]
y_balanced = y[balanced_idx]


indices = torch.randperm(len(X_balanced))
X_balanced = X_balanced[indices]
y_balanced = y_balanced[indices]

print(f"После class0: {(y_balanced==0).sum().item()}, class1: {(y_balanced==1).sum().item()}")