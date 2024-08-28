import torch
import torchvision
from tensorboard import summary
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

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

    def forward(self, image):
        output = self.module1(image)
        return output


loss = nn.CrossEntropyLoss()
model = DemoModule()
sgd = torch.optim.SGD(model.parameters(), lr=0.02)
writer=SummaryWriter("./logs")

for epoch in range(20):
    # 训练模式，会对某些特定层器作用，也可以不写
    # model.train()
    # 测试模式
    # model.eval()
    step = 0
    for data in dataloader:
        images, targets = data
        outputs = model(images)
        result_loss = loss(outputs, targets)
        sgd.zero_grad()
        result_loss.backward()
        sgd.step()
        step += 1
        if step % 100 == 0:
            writer.add_scalar("lose-2",result_loss.item(),step)
            print(result_loss)

