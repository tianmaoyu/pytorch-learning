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

# dataset = torchvision.datasets.CIFAR10("./dataset", train=False, download=True, transform=transforms.ToTensor())
# dataloader = DataLoader(dataset=dataset, batch_size=64)



def demo():
    loss = nn.MSELoss()

    input = torch.tensor([3],requires_grad=True, dtype=torch.float32)
    print(input)
    target = torch.tensor([5],dtype=torch.float32)
    print(target)

    output = loss(input, target)
    print(output)
    # backward 在loss 返回值中，（会设置 模型的梯度值）
    output.backward()
    print(output)


def demo_CrossEntropyLoss():
    loss = nn.CrossEntropyLoss()

    input = torch.tensor([0.1, 0.2, 0.3])
    target=torch.tensor([1])
    input= torch.reshape(input,(1,3))
    output=loss(input,target)
    print(output)

demo_CrossEntropyLoss()