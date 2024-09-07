import os
import torch
import matplotlib.pyplot as plt
import torchkeras
import torchmetrics.segmentation
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchkeras.vision import UNet
from torchkeras.metrics import IOU
from torchmetrics.classification import BinaryJaccardIndex,JaccardIndex
from torchkeras.vision import UNet

# train_dataset=torchvision.datasets.CIFAR10("./dataset",train=True,transform=torchvision.transforms.ToTensor())
# test_dataset=torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor())
#
# dl_train= DataLoader(dataset=train_dataset,batch_size=5)
# dl_val= DataLoader(dataset=test_dataset,batch_size=5)

transform = torchvision.transforms.ToTensor()
ds_train = torchvision.datasets.MNIST(root="./dataset",train=True,download=True,transform=transform)
ds_val = torchvision.datasets.MNIST(root="./dataset",train=False,download=True,transform=transform)
dl_train =  DataLoader(ds_train, batch_size=64, shuffle=True, num_workers=1)
dl_val =  DataLoader(ds_val, batch_size=64, shuffle=False, num_workers=1)

def create_net():
    net = nn.Sequential()
    net.add_module("conv1",nn.Conv2d(in_channels=1,out_channels=32,kernel_size = 3))
    net.add_module("pool1",nn.MaxPool2d(kernel_size = 2,stride = 2))
    net.add_module("conv2",nn.Conv2d(in_channels=32,out_channels=64,kernel_size = 5))
    net.add_module("pool2",nn.MaxPool2d(kernel_size = 2,stride = 2))
    net.add_module("dropout",nn.Dropout2d(p = 0.1))
    net.add_module("adaptive_pool",nn.AdaptiveMaxPool2d((1,1)))
    net.add_module("flatten",nn.Flatten())
    net.add_module("linear1",nn.Linear(64,32))
    net.add_module("relu",nn.ReLU())
    net.add_module("linear2",nn.Linear(32,10))
    return net


class Accuracy(nn.Module):
    def __init__(self):
        super().__init__()

        self.correct = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.total = nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        preds = preds.argmax(dim=-1)
        m = (preds == targets).sum()
        n = targets.shape[0]
        self.correct += m
        self.total += n

        return m / n

    def compute(self):
        return self.correct.float() / self.total

    def reset(self):
        self.correct -= self.correct
        self.total -= self.total


meanIou= torchmetrics.segmentation.MeanIoU(num_classes=2)

net = create_net()
model = torchkeras.KerasModel(net,
                              loss_fn = nn.BCEWithLogitsLoss(),
                              optimizer= torch.optim.Adam(net.parameters(),lr = 1e-4),
                              metrics_dict = {"acc":Accuracy()}
                             )
dfhistory=model.fit(train_data=dl_train,
                    val_data=dl_val,
                    epochs=5,
                    patience=3,
                    ckpt_path='checkpoint.pt',
                    monitor="val_meanIou",
                    mode="max",
                    plot=True
                   )

# # 创建 Binary Jaccard Index 指标实例
# metric = BinaryJaccardIndex()
# # 初始化一个列表来存储每次迭代的 IoU 值
# values = []
#
# # 生成并计算 10 次随机数据的 IoU 值
# for _ in range(10):
#     # 生成随机预测和标签
#     preds = torch.randn(10)  # 预测概率
#     target = torch.randint(low=0,high=2, size=(10,))  # 实际标签
#     # 将预测概率转换为二进制标签
#     preds_binary = (preds > 0.5).float()
#     # 更新并获取 IoU 值
#     value = metric(preds_binary, target)
#     values.append(value.item())
#
# # 绘制 IoU 值的变化曲线
# plt.figure(figsize=(10, 5))
# plt.plot(range(1, 11), values, marker='o')
# plt.title('IoU Values Over Iterations')
# plt.xlabel('Iteration')
# plt.ylabel('IoU Value')
# plt.grid(True)
# plt.show()
