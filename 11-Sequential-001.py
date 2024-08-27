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


def demo():
    module = nn.Sequential(
        nn.Conv2d(3, 32, 5,padding=2),
        nn.MaxPool2d(kernel_size=(2,2)),
        nn.Conv2d(32, 32, 5,padding=2),
        nn.MaxPool2d(kernel_size=(2, 2)),
        nn.Conv2d(32, 64, 5, padding=2),
        nn.MaxPool2d(kernel_size=(2, 2)),
        nn.Flatten(),
        nn.Linear(64*4*4,64),
        nn.Linear(64, 10)
    )
    input = torch.randn(64,3,32,32)
    output= module(input)
    print(output.shape)


demo()
