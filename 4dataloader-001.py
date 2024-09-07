from torch.utils.data import DataLoader, random_split
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms



# 数据拆分
test_data=torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor())
test_loader=DataLoader(dataset=test_data,batch_size=4,shuffle=True,num_workers=0,drop_last=False)
total_length=len(test_data)
val_length=int(total_length*0.1)
tarin,val=random_split(test_data, [total_length-val_length, val_length])



for epoch in range(2):
    for data in test_loader:
        images, labels=data



