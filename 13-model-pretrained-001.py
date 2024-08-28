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

dataset = torchvision.datasets.CIFAR10("./dataset", train=True, download=True, transform=transforms.ToTensor())
dataloader = DataLoader(dataset=dataset, batch_size=64)


class DemoModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.module1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, input):
        output = self.module1(input)
        return output


loss = nn.CrossEntropyLoss()
module = DemoModule()
sgd = torch.optim.SGD(module.parameters(),lr=0.02)


for epoch in range(20):
    step=0
    for data in dataloader:
        images, targets = data
        outputs = module(images)
        result_loss= loss(outputs,targets)
        sgd.zero_grad()
        result_loss.backward()
        sgd.step()

        if step%100==0:
            print(result_loss)
        step += step



