'''
    - gradient 계산법
    - 향후 최적화 단계에서 중요하게 작용됨
'''

import torch

# gradient 계산이 필요함을 추적함
x = torch.randn(3, requires_grad=True)
# print(x)

y = x + 2
# print(y)
z = y*y*2
z = z.mean()
# print(z)

# 스칼라 값에 대해서만 수행 가능 (단일 값)
# 스칼라 값(단일 값)에 backward 연산 수행 시, backward 매개변수에 벡터 인수 필요 없음
# 벡터 값 backward 연산 수행 시, backward 매개변수에 벡터 인수 필요 => 없을 시 error
# jacobian 연산
z.backward() # dz / dx
# print(x.grad)

v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)
z = y*y*2
z.backward(v)
# print(x.grad)

'''
    requires_grad=True 을 해지하는 방법
    1. x.requires_grad_(False)
    2. x.detach()
    3. with torch.no_grad():
'''
# x.requires_grad_(False)
# print(x)

# y = x.detach()
# print(y)

with torch.no_grad():
    y = x + 2
    # print(y)


# gradient 값이 축적되는 것을 방지해야함
weights = torch.ones(4, requires_grad=True)

for epoch in range(3):
    model_output = (weights * 3).sum()
    model_output.backward()
    print(weights.grad)

    # make gradient into 0 value for preventing accumulation
    weights.grad.zero_()

# optimizer example
optimizer = torch.optim.SGD(weights, lr=0.01)
optimizer.step()
optimizer.zero_grad()