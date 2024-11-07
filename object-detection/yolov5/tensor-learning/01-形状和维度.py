import torch
from PIL import Image
from sympy.stats.sampling.sample_numpy import numpy
from torchvision.transforms import ToTensor

scale= torch.tensor(1)
boolean= torch.tensor(False)
print(scale.shape,scale.ndimension())
# 不是 1 行两列， 维度=1
vector = torch.tensor([1, 2])
print(vector.shape,vector.ndimension())
matrix=torch.randn(size=[2,3])
# 一行两列， 维度=2
data3 = torch.tensor([[1,2]])
data4 = torch.tensor([[1],[2]])
three_data= torch.arange(27).reshape([3,3,3])
print(three_data.shape,three_data.ndimension())
print(three_data)

# 图片
image_path = "../data/0000001_05499_d_0000010.jpg"
image = Image.open(image_path).convert("RGB")
# height,width,channel
array= numpy.array(image)
# channel,height,width
tensor=ToTensor()(image)
print(tensor.shape,tensor.ndimension())

# 更好的理解
data=torch.randn([10,3,80,8],dtype=torch.float)