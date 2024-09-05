import torch
import torch.nn as nn
import torchvision.models as models


class BasicBlock(nn.Module):
    # 输出通道 扩张
    # expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._get_layer_1()
        self.layer2 = self._get_layer_2()
        self.layer3 = self._get_layer_3()
        self.layer4 = self._get_layer_4()

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512, num_classes)

    def _get_layer_1(self):
        blocks = []
        blocks.append(BasicBlock(64, 64, 1))
        blocks.append(BasicBlock(64, 64, 1))
        return nn.Sequential(*blocks)

    def _get_layer_2(self):
        blocks = []
        blocks.append(BasicBlock(64, 128, 2))
        blocks.append(BasicBlock(128, 128, 1))
        return nn.Sequential(*blocks)

    def _get_layer_3(self):
        blocks = []
        blocks.append(BasicBlock(128, 256, 2))
        blocks.append(BasicBlock(256, 256, 1))
        return nn.Sequential(*blocks)

    def _get_layer_3(self):
        blocks = []
        blocks.append(BasicBlock(256, 512, 2))
        blocks.append(BasicBlock(512, 512, 1))
        return nn.Sequential(*blocks)

    # def _make_layer(self, block, out_channels, stride):
    #     strides = [stride,1]
    #     layers = []
    #     # 保证每次 下采样，顺序正确
    #     for stride in strides:
    #         layers.append(block(self.in_channels, out_channels, stride))
    #         self.in_channels = out_channels
    #     return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def ResNet18(num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

# Example usage:
# model = ResNet18(num_classes=1000)
# print(model)
