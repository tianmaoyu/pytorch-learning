# github 标准的模型  https://github.com/tianmaoyu/Pytorch-UNet/tree/master/unet

import torch
from torch import nn
from torchvision import transforms
from torch.nn import functional as F
import data


class UnetDemo(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        def block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(inplace=True),

                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(inplace=True),
            )

        def down_simple(in_channels, out_channels):
            return nn.Sequential(
                nn.MaxPool2d(2),
                block(in_channels, out_channels)
            )

        def up_simple(in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
                block(in_channels, out_channels)
            )

        self.input = block(in_channel, 64)

        self.down1 = down_simple(64, 128)
        self.down2 = down_simple(128, 256)
        self.down3 = down_simple(256, 512)
        self.down4 = down_simple(512, 1024)

        self.center = block(1024, 1024)

        self.up1 = up_simple(128, 64)
        self.up2 = up_simple(256, 128)
        self.up3 = up_simple(512, 256)
        self.up4 = up_simple(1024, 512)

        self.output = nn.Conv2d(64, out_channel, kernel_size=1)

    def forward(self, x):
        input = self.input(x)
        down1 = self.down1(input)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)

        center = self.center(down4)

        up4 = self.up4(center)
        cat4 = torch.cat((up4, down4), dim=1)

        up3 = self.up3(cat4)
        cat3 = torch.cat((up3, down3), dim=1)

        up2 = self.up2(cat3)
        cat2 = torch.cat((up2, down2), dim=1)

        up1 = self.up1(cat2)
        cat1 = torch.cat((up1, down1), dim=1)

        return torch.sigmoid(self.final_conv(cat1))


if __name__ == '__main__':
    in_data = torch.randn(1, 3, 111, 555)
    # 防止
    in_data = data.pad_16(in_data)
    model = UnetDemo(3, 1)
    out_data = model(in_data)
    print(out_data.shape)
