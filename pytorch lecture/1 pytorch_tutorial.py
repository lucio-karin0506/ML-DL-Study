import torch
import numpy as np
from zmq import device

# 1. basic tensor conception
x = torch.empty(2, 3)
x = torch.rand(2, 2)
x = torch.zeros(2, 2)
x = torch.ones(2, 2, dtype=torch.float16)
x = torch.rand(4, 4)

# 2. automatically determine the row size if -1
y = x.view(-1, 8)
# print(x)
# print(y.size())

# 3. convert between numpy and torch
# a = torch.ones(5)
# print(a)
# b = a.numpy()
# print(b)

# gpu가 아닌 cpu환경에서 연산 시 두 변수 모두 같은 메모리 위치를 가리키기 때문에 하나의 값이 바뀌면 다른 하나의 값도 바뀜
# a.add_(1)
# print(a)
# print(b)

# gpu 위에서 수행할 시
if torch.cuda.is_available():
    device = torch.device('cuda')
    x = torch.ones(5, device=device)
    y = torch.ones(5)

    # 해당 텐서를 gpu 장치로 이동
    y = y.to(device)
    z = x + y

    # numpy는 cpu 텐서에서만 처리가능하므로 다시 cpu 장치로 이동
    z = z.to('cpu')
    z.numpy()

# 5. 최적화하고 싶은 변수가 있으면 해당 텐서 변수에 그래디언트가 필요함을 알려줘야함
x = torch.ones(5, requires_grad=True)
print(x)