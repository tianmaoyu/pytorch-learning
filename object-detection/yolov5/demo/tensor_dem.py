import torch
data0= torch.tensor(1)
data01= torch.tensor(False)
data1=torch.randn(size=[2,3])
# 不是 1 行两列， 维度=1
data2 = torch.tensor([1, 2])
# 一行两列， 维度=2
data3 = torch.tensor([[1,2]])
data4 = torch.tensor([[1],[2]])
print(data1)