import torch

def act(x):
    return 0 if x < 0.5 else 1

def go(hs, rk, at):
    X = torch.tensor([hs, rk, at], dtype=torch.float32)
    Wh = torch.tensor([[0.3, 0.3, 0], [0.4, -0.5, 1]])
    Wout = torch.tensor([-1., 1.])

    Zh = torch.mv(Wh, X) # умножение матрицы весов на входной вектор (матрица * вектор)
    """
    z1 = X[0] * Wh[0, 0] + X[1] * Wh[0, 1] + X[2] * Wh[0, 2]
    z2 = X[0] * Wh[1, 0] + X[1] * Wh[1, 1] + X[2] * Wh[1, 2]
    """
    print(f"значение сумм на нейронах скрытого слоя: {Zh}")

    Uh = torch.tensor([act(x) for x in Zh], dtype=torch.float32)
    print(f"значения на выходах нейронов скрытого слоя: {Uh}")

    Zout = torch.dot(Wout, Uh) # скалярное произведение
    """
    Wout[0] * Uh[0] + Wout[1] * Uh[1]
    """
    Y = act(Zout)
    print(f"Выходное значение НС: {Y}")

    return Y

if __name__ == "__main__":
    hs = 1
    rk = 0
    at = 1

    r = go(hs, rk, at)
    if r == 1:
        print("ты мне нравишься")
    else:
        print("созвонимся")