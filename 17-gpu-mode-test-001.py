import os

import torch
import torchvision
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
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

    def forward(self, image):
        output = self.module1(image)
        return output

dataset = torchvision.datasets.CIFAR10("./dataset", train=True, download=True, transform=transforms.ToTensor())

model = torch.load("./dataset/my-model-99.pth",map_location=torch.device('cpu'))

image = Image.open("data/test.jpg")

preprocess=transforms.Compose([
transforms.Resize((32,32)),
transforms.ToTensor(),
])
image = preprocess(image)

image=torch.reshape(image,(1,3,32,32))


with torch.no_grad():
    output= model(image)

print(output)
print(dataset.class_to_idx)