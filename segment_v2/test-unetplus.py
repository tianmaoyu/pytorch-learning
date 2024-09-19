import PIL.ImageShow
import torch
import torchvision.utils
from torch import nn
from  PIL import  Image
from torchvision.transforms import ToTensor
import  utils
from  unetplus import  UNetPlus,DoubleConv


model_path= "src/unetplus/unetplus-4.pth"
image_path="./src/DJI_20240423132533_0004_Z.JPG"

image = Image.open(image_path).convert("RGB")
model= torch.load(model_path,map_location="cpu")
# model= torch.load(model_path,weights_only=False)
# model=UNetV2(3,1)
# model.load_state_dict(dic)

with torch.no_grad():

    image=ToTensor()(image)
    image= image.unsqueeze(dim=0)
    image = utils.pad_16(image)
    # image = image.to('cuda')
    output= model(image)
    torchvision.utils.save_image(output,fp="./src/unetplus-z-4.jpg")
    print("---"*8)


