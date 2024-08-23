import torch
from torch import nn
import  torch.nn.functional as F


class HelloWorld(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input + 1


input = torch.tensor([
    [1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5],
],dtype=float)
kernel = torch.tensor([
    [0.5, 0.5, 0.5],
    [0.5, 1, 0.5],
    [0.5, 0.5, 0.5],
],dtype=float)


#  四维向量，也可以 理解为 batch size 样本数量，  channels 表示通道数， 最后的 5，5 表示 高宽
input=torch.reshape(input,(1,1,5,5))
kernel=torch.reshape(kernel,(1,1,3,3))
obj = F.conv2d(input, kernel, stride=1,padding=1)
print(obj)



# filters = torch.randn(3, 3)
# inputs = torch.randn(5, 5)
# 卷积操作
conv1 = nn.Conv2d(1, 20, 5)
# obj = nn.Conv2d(inputs, filters, padding=1)

hello_world = HelloWorld()
input_data = torch.tensor(1)
output_data = hello_world(input_data)
print(output_data)
