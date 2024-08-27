import numpy
import torch
import torchvision.datasets
from torch import nn
import torch.nn.functional as F
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision import transforms
# import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.pyplot as plt

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, download=True, transform=transforms.ToTensor())
dataloader = DataLoader(dataset=dataset, batch_size=64)

class DemoModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))

    def forward(self, input):
        output = self.maxpool1(input)
        return output


# module = DemoModule()
#
# writer = SummaryWriter("./logs")
#
# step = 1
# for data in dataloader:
#     input,targets=data
#     output = module(input)
#     writer.add_images("input", input,step)
#     writer.add_images("output", output,step)
#     step += step
#     if (step > 10):
#         break


def demo():
    img,target= dataset[0]
    # transforms.ToPILImage()()
    # 调换 axes
    plt.imshow(numpy.transpose(img, (1, 2, 0)))
    plt.show()
    # 图像展平
    # torch.reshape()
    input= torch.flatten(img)
    linear = nn.Linear(3*32*32, 10,bias=True)

    output = linear(input)
    print(output.size())



demo()