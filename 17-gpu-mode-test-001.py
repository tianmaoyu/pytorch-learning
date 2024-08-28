import torch
import torchvision
from PIL import Image
# from tensorboard import summary
from torch import nn
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

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


# device=torch.device("gpu:0")
loss = nn.CrossEntropyLoss()
model = DemoModule()

input_image = Image.open("data/test.jpg")


input_tensor = preprocess(input_image)
input_tensor = input_tensor.reshape(1, *input_tensor.shape)
