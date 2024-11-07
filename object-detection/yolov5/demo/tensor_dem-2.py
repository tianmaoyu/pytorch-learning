import torch

x = torch.tensor([[1, 1], [2, 2]])
y = torch.tensor([[3, 3], [4, 4]])
stacked = torch.stack([x, y], dim=2)
print(stacked.shape)
