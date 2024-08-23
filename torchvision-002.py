import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

transform_pipeline=transforms.Compose([
    transforms.ToTensor()
])
train_data=torchvision.datasets.CIFAR10(root="./dataset",train=True,transform=transform_pipeline, download=True)


writer = SummaryWriter("logs")

for i in range(10):
   image,label=  train_data[i]
   writer.add_image("dataset",image,i)

writer.close()