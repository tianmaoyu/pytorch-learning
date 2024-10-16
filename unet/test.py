import PIL.ImageShow
import torch
import torchvision.utils
from torch import nn
from  PIL import  Image
from torchvision.transforms import ToTensor

from unet import utils

model_path="./dataset/unet-99.pth"
image_path="./dataset/test.jpg"
image= utils.keep_image_size(image_path)

model= torch.load(model_path,map_location="cpu",weights_only=False)

with torch.no_grad():

    image=ToTensor()(image)
    image= image.unsqueeze(dim=0)

    output= model(image)
    torchvision.utils.save_image(output,fp="./out/test-99.jpg")


