import torch

t1 = torch.arange(100000).reshape(250, -1).float()
print(t1)

mask = torch.rand_like(t1)
print(mask)

count = 0

mask = mask.reshape(100000)
print(mask)

print((mask < 0.1).sum())

if 0:
    print('=============')
