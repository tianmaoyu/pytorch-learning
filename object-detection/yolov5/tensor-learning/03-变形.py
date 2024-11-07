import  torch
from PIL import Image
import numpy

data= torch.arange(27).reshape(3,3,3)
print(data)
data_view=data.view(3,9)
print(data_view)
data_permute=data_view.permute(1,0)
print(data_permute)

data1= torch.arange(40).reshape(2,5,2,2)
print(data1)
data_permute1=data1.permute(0,2,3,1)
print(data_permute1)

# 内存重新排列
# data_contiguous=data_permute.contiguous()
# print(data_contiguous)

layer= torch.randn([1,3*(80+5),80,80])
data= layer.view([1,3,85,80,80]).permute(0,1,3,4,2).contiguous()


# 图片
image_path = "../data/0000022_00500_d_0000005.jpg"
image = Image.open(image_path).convert("RGB")
# height,width,channel
array= numpy.array(image)
image_tensor= torch.tensor(array)
image_tensor1=image_tensor.permute(2,0,1)
print(image_tensor1)