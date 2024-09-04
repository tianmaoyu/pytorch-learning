import PIL.ImageShow
import torch
import torchvision.utils
from torch import nn
from  PIL import  Image
from torchvision.transforms import ToTensor
import  utils
from  net_standard_v1 import UnetDemoV1,DoubleConv,UpSimple,DownSimple


model_path="./water/unet-18.pth"
image_path="./water/DJI_20240423132549_0005_W.JPG"

image = Image.open(image_path).convert("RGB")
model= torch.load(model_path,map_location="cpu",weights_only=False)
# model= torch.load(model_path,weights_only=False)
# model=UNetV2(3,1)
# model.load_state_dict(dic)

with torch.no_grad():

    image=ToTensor()(image)
    image= image.unsqueeze(dim=0)
    image = utils.pad_16(image)
    # image = image.to('cuda')
    output= model(image)
    torchvision.utils.save_image(output,fp="./water/unet-18-5.jpg")
    print("---"*8)


