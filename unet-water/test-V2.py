import PIL.ImageShow
import torch
import torchvision.utils
from torch import nn
from  PIL import  Image
from torchvision.transforms import ToTensor
import  utils
from  net_standard_v2 import UNetV2


model_path="./water/water_bodies_model.pth"
image_path="./water/DJI_20240423132515_0002_W.JPG"

image = Image.open(image_path).convert("RGB")
dic= torch.load(model_path,map_location="cpu")
# model= torch.load(model_path,weights_only=False)
model=UNetV2(3,1)
model.load_state_dict(dic)

with torch.no_grad():

    image=ToTensor()(image)
    image= image.unsqueeze(dim=0)
    image = utils.pad_16(image)
    # image = image.to('cuda')
    output= model(image)
    torchvision.utils.save_image(output,fp="./water/water_bodies_model.jpg")
    print("---"*8)


