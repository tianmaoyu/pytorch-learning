import PIL.ImageShow
import torch
import torchvision.utils
from torch import nn
from  PIL import  Image
from torchvision.transforms import ToTensor
import  utils
from  unet import  Unet,DoubleConv,DownSimple,UpSimple
from torchvision.models import AlexNet

model_path= "src/infrared_unet/unet-10.pth"
image_path="./src/DJI_20240822122651_0027_T.JPG"
# image_path="./src/DJI_20240918150844_0011_T.JPG"

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
    torchvision.utils.save_image(output,fp="./src/infrared_unet/027-10.jpg")
    # torchvision.utils.save_image(output,fp="./src/infrared_unet/027-9.jpg")
    print("---"*8)


