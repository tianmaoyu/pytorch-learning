import torch
import torch.nn as nn
import torch.nn.functional as F

# 创建一个简单的输入张量，假设这是一个单通道的图像
input = torch.randn(1, 1, 10, 10)

conv=nn.Conv2d(in_channels=1,out_channels=3,kernel_size=7,stride=2,padding=3,bias=False)
output=conv(input)
print(output)



# 定义一个卷积层，没有使用膨胀率
conv_no_dilation = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, dilation=1)
output_no_dilation = conv_no_dilation(input)

# 定义一个卷积层，使用了膨胀率为 2
conv_with_dilation = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, dilation=2,padding=2)
output_with_dilation = conv_with_dilation(input)

print("Output without dilation:")
print(output_no_dilation)
print("\nOutput with dilation rate of 2:")
print(output_with_dilation)