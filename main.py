# первый пример НС
import torch

X = torch.tensor([1.0, 2.0, 3.0, 4.0]) 
Y = torch.tensor([5.0, 12.0, 15.0, 21.0])

W = torch.tensor([10.0], requires_grad=True) 

print(f"Стартовый вес (догадка компьютера): {W.item():.2f}\n")

learning_rate = 0.01 

for step in range(10000):
    
    y_pred = 5 * X
    
    loss = torch.mean((y_pred - Y) ** 2)
    loss.backward() 
    
    with torch.no_grad():
        W -= learning_rate * W.grad  
        
        W.grad.zero_()
    
    if step % 1000 == 0:
        print(f"Шаг {step}: Вес = {W.item():.4f}, Ошибка = {loss.item():.4f}")

print(f"\nИтоговый вес: {W.item():.4f} (Правильный ответ: 5.0)")
