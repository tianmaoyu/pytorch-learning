import PIL.ImageShow
import torch
import torchvision.utils
from torch import nn
from  PIL import  Image
from torchvision.transforms import ToTensor
import  utils

# 退出条件，三次小于最佳的均值，
test_list=[1,2,3,5,6,10.9,7,8,8,8,5,5,6,5]
list=[0]
less_count=0
best=0
for item in test_list:
    total= sum(list[-3:])
    if total> best:
        best= total
    if best/3 <= item:
        print(list)
        print("继续")
        list.append(item)
        print(list)
    else:
        less_count+=1
        if(less_count>3):
            print("退出")
            break
        print(list)



model_path="./src/PSPNet-8.pth"
image_path="./src/DJI_20240423132515_0002_W.JPG"

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
    torchvision.utils.save_image(output,fp="./water/2-8.jpg")
    print("---"*8)


