import torch 

A = torch.randn(2, 3, 4, 5)
print(A)

B = A.flatten() # но можно было бы использовать 'ravel()' (но flatten возвращает всегда копию)
print(B)

B = B.reshape_as(A)
print(B)

print(torch.equal(A, B)) # True
