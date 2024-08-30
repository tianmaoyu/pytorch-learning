import torch
from torch import nn

from PIL import  Image
#
def conv_transpose_demo():
    conv_transpose = nn.ConvTranspose2d(1024, 512, 2, 2)
    input_data = torch.randn(1, 1024, 100, 100)
    output_data = conv_transpose(input_data)
    print(output_data.shape)


conv= nn.Conv2d(3 ,30 ,kernel_size=3 ,padding=1 ,padding_mode="reflect" ,bias=False)
input_data= torch.rand(1,3,100,100)
output_data=conv(input_data)
print(output_data.shape)

pool= nn.MaxPool2d(kernel_size=2,stride=2)

output_data=pool(output_data)
print(output_data.shape)