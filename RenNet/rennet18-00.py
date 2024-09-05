import torch
import torch.nn as nn

class ResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet18, self).__init__()

        # 输入层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # layer 1
        self.conv2_1_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2_1_bn1 = nn.BatchNorm2d(64)
        self.conv2_1_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2_1_bn2 = nn.BatchNorm2d(64)

        self.conv2_2_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2_2_bn1 = nn.BatchNorm2d(64)
        self.conv2_2_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2_2_bn2 = nn.BatchNorm2d(64)

        # layer 2
        self.conv3_1_conv1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv3_1_bn1 = nn.BatchNorm2d(128)
        self.conv3_1_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3_1_bn2 = nn.BatchNorm2d(128)
        self.conv3_1_downsample = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(128)
        )

        self.conv3_2_conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3_2_bn1 = nn.BatchNorm2d(128)
        self.conv3_2_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3_2_bn2 = nn.BatchNorm2d(128)

        # layer 3
        self.conv4_1_conv1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv4_1_bn1 = nn.BatchNorm2d(256)
        self.conv4_1_conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4_1_bn2 = nn.BatchNorm2d(256)
        self.conv4_1_downsample = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(256)
        )

        self.conv4_2_conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4_2_bn1 = nn.BatchNorm2d(256)
        self.conv4_2_conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4_2_bn2 = nn.BatchNorm2d(256)

        # layer 4
        self.conv5_1_conv1 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv5_1_bn1 = nn.BatchNorm2d(512)
        self.conv5_1_conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv5_1_bn2 = nn.BatchNorm2d(512)
        self.conv5_1_downsample = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(512)
        )

        self.conv5_2_conv1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv5_2_bn1 = nn.BatchNorm2d(512)
        self.conv5_2_conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv5_2_bn2 = nn.BatchNorm2d(512)

        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # 输入层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # layer 1
        identity = x
        x = self.conv2_1_conv1(x)
        x = self.conv2_1_bn1(x)
        x = self.relu(x)
        x = self.conv2_1_conv2(x)
        x = self.conv2_1_bn2(x)
        x += identity
        x = self.relu(x)

        identity = x
        x = self.conv2_2_conv1(x)
        x = self.conv2_2_bn1(x)
        x = self.relu(x)
        x = self.conv2_2_conv2(x)
        x = self.conv2_2_bn2(x)
        x += identity
        x = self.relu(x)

        # layer 2
        identity = self.conv3_1_downsample(x)
        x = self.conv3_1_conv1(x)
        x = self.conv3_1_bn1(x)
        x = self.relu(x)
        x = self.conv3_1_conv2(x)
        x = self.conv3_1_bn2(x)
        x += identity
        x = self.relu(x)

        identity = x
        x = self.conv3_2_conv1(x)
        x = self.conv3_2_bn1(x)
        x = self.relu(x)
        x = self.conv3_2_conv2(x)
        x = self.conv3_2_bn2(x)
        x += identity
        x = self.relu(x)

        # layer 3
        identity = self.conv4_1_downsample(x)
        x = self.conv4_1_conv1(x)
        x = self.conv4_1_bn1(x)
        x = self.relu(x)
        x = self.conv4_1_conv2(x)
        x = self.conv4_1_bn2(x)
        x += identity
        x = self.relu(x)

        identity = x
        x = self.conv4_2_conv1(x)
        x = self.conv4_2_bn1(x)
        x = self.relu(x)
        x = self.conv4_2_conv2(x)
        x = self.conv4_2_bn2(x)
        x += identity
        x = self.relu(x)

        # layer 4
        identity = self.conv5_1_downsample(x)
        x = self.conv5_1_conv1(x)
        x = self.conv5_1_bn1(x)
        x = self.relu(x)
        x = self.conv5_1_conv2(x)
        x = self.conv5_1_bn2(x)
        x += identity
        x = self.relu(x)

        identity = x
        x = self.conv5_2_conv1(x)
        x = self.conv5_2_bn1(x)
        x = self.relu(x)
        x = self.conv5_2_conv2(x)
        x = self.conv5_2_bn2(x)
        x += identity
        x = self.relu(x)

        # 全局平均池化
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# 创建模型实例
model = ResNet18(num_classes=1000)
print(model)