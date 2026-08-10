import torch 

X = torch.rand(1200, 2)*4-2 # -2, 2
z = (X[:, 0]**2 + X[:, 1]**2 + torch.randn(1200)).unsqueeze(-1)

percent_test_set = 20
size_train = int(X.shape[0] - X.shape[0] * percent_test_set / 100)

X_train = X[:size_train]
z_train = z[:size_train]

X_test = X[size_train:]
z_test = z[size_train:]

print(X_train.shape)
print(X_test.shape)

print(z_train.shape)
print(z_test.shape)