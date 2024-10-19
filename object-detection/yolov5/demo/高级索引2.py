import torch

data=torch.tensor([[1.25000, 1.62500],
                   [2.00000, 3.75000],
                   [4.12500, 2.87500]])

index=torch.tensor([1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2])

out1= data[index]
out2= data[[1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2]]

print(torch.equal(out1,out2))
