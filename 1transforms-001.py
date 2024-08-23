from torchvision import transforms
from PIL import Image
import numpy
from  torch import  Tensor

image=Image.open("data/train/ants/kurokusa.jpg")
image_np=numpy.array(image)
to_tensor = transforms.ToTensor()
tensor_image = to_tensor(image_np)
print(image_np.shape)
print(tensor_image)

# 均值和标准差
trans_norm=transforms.Normalize([0.5,0.5,0.5],[0.5,0.3,0.4])
trans_resize=transforms.Resize((100,100))
image_resize=trans_resize(tensor_image)
print(image_resize)

obj = transforms.Compose()
