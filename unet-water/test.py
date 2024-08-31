import PIL.ImageShow
import torch
import torchvision.utils
from torch import nn
from  PIL import  Image
from torchvision.transforms import ToTensor
import  utils


model_path="./water/unet-86.pth"
image_path="./water/2.png"

image = Image.open(image_path).convert("RGB")
# model= torch.load(model_path,map_location="cpu",weights_only=False)
model= torch.load(model_path,weights_only=False)

with torch.no_grad():

    image=ToTensor()(image)
    image= image.unsqueeze(dim=0)
    image = utils.pad_16(image)
    image = image.to('cuda')
    output= model(image)
    torchvision.utils.save_image(output,fp="./water/2-86-cuda.jpg")
    print("---"*8)


