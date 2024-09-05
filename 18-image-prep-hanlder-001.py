import os

import torch
import torchvision
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F


# 创建一个简单的二维张量
# x = torch.tensor([[1, 2], [3, 4]])
x = torch.randn(1,1,2,2)
# 在左边填充2个元素，在右边填充1个元素，在顶部填充1个元素，在底部填充2个元素
padded_x = F.pad(x, (0, 1, 1, 2), mode='constant',value=0)
print(padded_x)