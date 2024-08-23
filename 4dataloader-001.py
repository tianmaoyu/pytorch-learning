from torch.utils.data import DataLoader
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


test_data=torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor())
test_loader=DataLoader(dataset=test_data,batch_size=4,shuffle=True,num_workers=0,drop_last=False)

for epoch in range(2):
    for data in test_loader:
        images, labels=data
