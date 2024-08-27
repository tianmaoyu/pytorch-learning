import torch
from torch import nn
import  torch.nn.functional as F
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision import transforms

data = datasets.CIFAR10("./dataset", train=False,transform=transforms.ToTensor())
dataloader=DataLoader(data,batch_size=64)
class HelloWorld(nn.Module):
    def __init__(self):
        super(HelloWorld,self).__init__()
        self.conv1=Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=1)

    def forward(self, input):
        output=self.conv1(input)
        return output


hello_world = HelloWorld()
writer = SummaryWriter("logs")
setp=0
for data in dataloader:
    images,targets=data
    output_imgs= hello_world(images)
    writer.add_images("images",images,global_step=setp)

    output_imgs=torch.reshape(output_imgs,(-1,3,32,32))
    writer.add_images("output_imgs", output_imgs, global_step=setp)
    setp+=1
    if setp>10:
        break

