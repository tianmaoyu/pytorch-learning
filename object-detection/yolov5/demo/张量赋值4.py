import numpy as np
import torch


print(torch.tensor(True))

target = torch.zeros([1, 3, 20, 20, 85])
mask = torch.zeros([1, 3, 20, 20, 85], dtype=torch.bool)

target[0,0,0,0,:4] =torch.tensor([1, 2, 3, 4])
print(target)