import PIL.ImageShow
import torch
import torchvision.utils
from torch import nn
from  PIL import  Image
from torchvision.transforms import ToTensor

from unet import utils

# https://modelscope.cn/my/mynotebook/preset
model_path="./water/unet-234.pth"
image_path="./water/DJI_20240423132515_0002_W.JPG"
image_path="./img/DJI_20230630102813_0046_W.JPG"
image= utils.keep_image_size(image_path)

model= torch.load(model_path,map_location="cpu",weights_only=False)

with torch.no_grad():

    image=ToTensor()(image)
    image= image.unsqueeze(dim=0)

    output= model(image)
    torchvision.utils.save_image(output,fp="./img/test-234.jpg")


