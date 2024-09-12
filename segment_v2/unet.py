# github 标准的模型  https://github.com/tianmaoyu/Pytorch-UNet/tree/master/unet

import torch
from torch import nn
import utils


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.layer(x)


class DownSimple(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.layer(x)


class UpSimple(nn.Module):
    def __init__(self, in_channels,out_channels):
        super().__init__()
        # 上采样 channel 减半
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # 卷积 channel 减半
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, up_x, cat_left_x):
        upx = self.up(up_x)
        # 注意 链接时左右
        x = torch.cat((cat_left_x, upx), dim=1)
        return self.double_conv(x)


class Unet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.input = DoubleConv(in_channel, 64)

        self.down1 = DownSimple(64, 128)
        self.down2 = DownSimple(128, 256)
        self.down3 = DownSimple(256, 512)
        self.down4 = DownSimple(512, 1024)

        self.up4 = UpSimple(1024,512)
        self.up3 = UpSimple(512,256)
        self.up2 = UpSimple(256,128)
        self.up1 = UpSimple(128,64)

        self.output = nn.Conv2d(64, out_channel, kernel_size=1)

    def forward(self, x):
        x1 = self.input(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        up4 = self.up4(x5,x4)
        up3 = self.up3(up4,x3)
        up2 = self.up2(up3,x2)
        up1 = self.up1(up2,x1)
        output= self.output(up1)

        return torch.sigmoid(output)


if __name__ == '__main__':
    in_data = torch.randn(1, 3, 111, 555)
    # 防止
    in_data = utils.pad_16(in_data)
    model = Unet(3, 1)
    out_data = model(in_data)
    print(out_data.shape)
