import torch
import numpy as np

if __name__ == "__main__":
    dp = np.array([1,2,3])

    t = torch.from_numpy(dp) # dp зависит от t (и наоборот)
    print(t)
    print(dp)

    dp[0] = 10
    print(t)
    print(dp)

    t2 = torch.tensor(dp, dtype=torch.float32) # dp не зависит от t2 (и наоборот)
    print(t2)

    # Автозаполнение и изменение формы 

    print(torch.zeros(2,3)) # по умолчанию тип элементов равен float32
    print(torch.zeros(2,2, dtype=torch.int32))
    print(torch.ones(2,3, dtype=torch.long)) # единицы
    print(torch.eye(3,3)) # по диагонали единицы
    print(torch.eye(3,2, dtype=torch.int16))
    print(torch.arange(7)) # арифмитеская прогрессия с шагом 1 от 0 до 7 (не включительно) - синоним range(7)
    print(torch.arange(-5, 0))
    print(torch.arange(-5, 0, 2)) 
    print(torch.arange(1, 0, -0.2))

    print(torch.linspace(1,5, 2))
    print(torch.linspace(1,5, 1))
    print(torch.linspace(1,5, 3))
    print(torch.linspace(1,5, 4))

    print(torch.rand(2,3)) # сучайные значения в диапозоне от 0 до 1
    print(torch.randn(2,3))
    # print(torch.manual_seed(1)) - чтобы при каждом запуске не было постоянной энтропии (один раз рандомно а потом нет)


    # виды методов:
    # имя_ - inplace (mutable) методы, изменяющие текущий тензор, не создают нового;
    # имя  - immutable методы не меняют текущий тензор, формируют новый.   

    # пример "имя_"
    t = torch.FloatTensor(2,5)  
    t.fill_(-0.3)
    print(t)
    t.random_(1,7) # рандомные значения от 1 до 7 
    print(t)
    x = torch.FloatTensor(2,4).fill_(10)
    print(x)

    # разное предствление одного и того же тензора 
    x1 = torch.arange(27)
    x2 = x1.view(3,9) # тепреь x2 рассматривает тензор x1 по другому
    print(f"x1 = {x1}")
    print(f"x2 = {x2}") # итого; x1 и x2 представляют тензор по разному но ссылаются на одни и те же данные!
    x2[0] = 100 # у обоих тензоров изменилось значение по индексу т.к. ссылаются они на одни и те же данные! (еще раз)

    x1.resize_(2,3) # создание новое представление 
    print(x2)

    print(x1.ravel()) # представить многомерный тензор как одномерный
    
    # добавление и удаление оси 
    x = torch.arange(32).view(8,2,2)
    print(x.size()) # показать сколько осей
    x = torch.unsqueeze(x, dim=0) # добавление ПЕРВОЙ оси (нового пространства) 
    print(x.size()) 
    r = x.unsqueeze(0)
    print(r.size())

    x.unsqueeze_(0)
    print(x.size())

    b = torch.unsqueeze(x, dim=-1) # добавление ПОСЛЕДНЕЙ (-1) оси
    print(b.size())

    # удаление оси - squeeze()


    # индексирование и срезы
    x = torch.IntTensor([(1,2,3), (10, 20, 30), (100, 200, 300)])
    print(x)
    print(x[1,1])
    print(x[-1, -1])
    print(x[0])
    print(x[0, :])
    print(x[:, 1]) # где 1 - это ВТОРОЙ столбец матрицы
    

    # Базовые математически операции (+, -, *, /, //, **, %)

    a = torch.FloatTensor([1, 2, 3])
    print(a)
    print(a-3) # от всех значений просто вычитается число 3
    print(-a) # всем значениям просто присваивается МИНУС
    print(2 + a)
    print(a * 5)
    print(a / 5)
    print(a ** 3)
    print(a // 2)
    print(a % 2)
    
    b = torch.IntTensor([3, 4, 5])
    print(f"tensor a = {a}")
    print(f"tensor b = {b}")
    print(a - b) # будет вещественный тип потому что вещ тип более общий по отношению к целочисленному
    print(b + a)
    print(a * b)
    print(a / b)
    print(b // a)
    print(b ** a)
    print(b % a) # все операции работают ПО элементно 

    b = torch.IntTensor([3, 4, 5, 6])
    # a + b = > ошибка; тензор "b" не совпадает с длиной тензором "a"
    b = torch.arange(1, 7).view(2, 3)
    print(a)
    print(b)
    print(a + b) # транслирование тензоров (тензор "a" будет применен к КАЖДОЙ строке тензора "b") 

    a = torch.arange(1, 19).view(3, 3, 2)
    b = torch.ones(3, 2)
    print(a)
    print(b)
    print(a - b) # транслирование 
    print(a * 10) # число 10 это как бы тоже тензор состоящий из одного элемента и он как бы протранслирован на каждый элемент тензрора "a"

    a = torch.IntTensor([1, 2, 6, 8])
    print(a)

    a += 5
    print(a)
    b = torch.ones(4)
    print(b)
    b *= a
    print(b)

    # a += b - > запрещено производить операции с ..= (/=, //=, *=, +=, -= и тд) т. к. они имеют разные типы данных

    
    # Тригонометрические и статистические функции

    a = torch.FloatTensor([1, 2, 3, 10, 20, 30])
    print(a)
    print(a.sum())
    value = a.sum().item()
    print(value) # чтобы получить именно ЗНАЧЕНИЕ а не тензор 
    print(a.mean().item())
    print(a.max().item())

    print(a.view(3, 2).sum()) # метод применяется ко всем элементам несмотря на его представление  
    a = a.view(3, 2)
    print(a.sum(dim=0)) # суммирование по столбцам
    print(a.sum(dim=1)) # суммирование по строкам
    print(a.mean(dim=0)) 
    print(a.mean(dim=1)) 

    a = torch.IntTensor([-1, 1, 5, -44, 32, 2])
    print(a)

    print(torch.abs(a)) # если хочется изменить САМ тензор - "abs_" (с подчеркиванием) 
    print(torch.amax(a))
    print(torch.log(a))
    print(torch.round(a))
    a = a.float()
    print(a)
    torch.round_(a)
    print(a)

    print(torch.sin(a))

    # Векторно-матричные операции

    a = torch.arange(1, 10).view(3, 3)
    b = torch.arange(10, 19).view(3, 3)
    print(a)
    print(b)

    r1 = a * b # или torch.mul(a, b)
    print(r1) # -> это просто обычное умножение "тензоров"

    c = torch.matmul(a, b) # математическое умножение матриц! (по правилу)
    print(c)
    c = torch.mm(a, b) # без возможности транслирования (в отличии от .matmul()) 
    print(c)

    v = torch.LongTensor([-1, -2, -3])
    c = torch.matmul(a, v) # умножение вектора на матрицу
    print(c)

    # c = torch.mm(v, a) - > ошибка; поскольку размеры матриц разные - и транслирования НЕТ
    c = a.mm(b)
    c = a.matmul(b) # тоже самое что torch.matmul / torch.mm

    bx = torch.randn(7, 3, 5)
    by = torch.randn(7, 5, 4)
    print(bx)
    print(by)

    print(torch.bmm(bx, by))

    a = torch.arange(1, 10, dtype=torch.float32)
    b = torch.ones(9)

    print(a)
    print(b)

    c = torch.dot(a,b) # скалярное произведение матриц
    print(c)

    c = torch.outer(a, b) # внешнее произведение 
    print(c)

    a = torch.FloatTensor([1, 2, 3])
    b = torch.arange(4, 10, dtype=torch.float32).view(2, 3)
    print(a)
    print(b)

    # первым аргументом ОБЯЗАТЕЛЬНО должна идти МАТРИЦА! (иначе будет ошибка)
    print(torch.mv(b, a)) # умножение матрицы на вектор (по правилу которое ты знаешь)
    print(b.mv(a)) # или так 

    # элементы линейной алгебры

    """
    { 1x + 2x + 3x  = 10 (где коэфициенты к x это матрица A а то что равно это матрица Y)
    { 1x + 4x + 9x  = 20
    { 1x + 8x + 27x = 30
    """
    a = torch.FloatTensor([(1, 2, 3), (1, 4, 9), (1, 8, 27)])
    print(a)

    # вычисление ранга (чтоб быть уверенным что он состоит из линейно независимых строк и столбцов)
    print(torch.linalg.matrix_rank(a)) # на выходе 3 - значит что матрица способна описывать систему ИЗ трех НЕЗАВИСИМЫХ уравнений 
    y = torch.FloatTensor([10, 20, 30])
    print(torch.linalg.solve(a, y)) # решение системы уравнений (где определены коэфициенты в матрице A и сами Y (игрики))
    """"
    другой способ решения ЭТОЙ же системы уравнений через вычисление обратной матрицы 

    A*x = y (где A это матрицы состоящая из коэфициентов (матрица A) и y это вектор Y)
    x = A**-1 * y

    реализация;
    """
    invA = torch.linalg.inv(a) # вычисление обратной матрицы (для вектора A)
    x = torch.mv(invA, y) # все по формуле 
    print(x)
    # остальные функции из ЛИНАЛ'а см. в документации

    

    