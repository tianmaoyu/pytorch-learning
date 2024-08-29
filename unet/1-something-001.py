import torch
from torch import nn

from PIL import  Image
#
def conv_transpose_demo():
    conv_transpose = nn.ConvTranspose2d(1024, 512, 2, 2)
    input_data = torch.randn(1, 1024, 100, 100)
    output_data = conv_transpose(input_data)
    print(output_data.shape)
