import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layer(x)


class UNet3Plus(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.cov00 = DoubleConv(in_channels, 64)
        self.cov10 = DoubleConv(64, 128)
        self.cov20 = DoubleConv(128, 256)
        self.cov30 = DoubleConv(256, 512)
        self.cov40 = DoubleConv(512, 1024)

        self.cov01 = DoubleConv(64 + 128, 64)
        self.cov11 = DoubleConv(128 + 256, 128)
        self.cov21 = DoubleConv(256 + 512, 256)
        self.cov31 = DoubleConv(512 + 1024, 512)

        self.cov02 = DoubleConv(64 + 64 + 128, 64)
        self.cov12 = DoubleConv(128 + 128 + 256, 128)
        self.cov22 = DoubleConv(256 + 256 + 512, 256)

        self.cov03 = DoubleConv(64 + 64 + 64 + 128, 64)
        self.cov13 = DoubleConv(128 + 128 + 128 + 256, 128)

        self.cov04 = DoubleConv(64 + 64 + 64 + 64 + 128, 64)

        self.final = nn.Conv2d(64, out_channels,kernel_size=1)

    def forward(self, x):
        x00 = self.cov00(x)
        x10 = self.cov10(self.down(x00))
        x01 = self.cov01(torch.cat([x00, self.up(x10)], dim=1))

        x20 = self.cov20(self.down(x10))
        x11 = self.cov11(torch.cat([x10, self.up(x20)], dim=1))
        x02 = self.cov02(torch.cat([x00, x01, self.up(x11)], dim=1))

        x30 = self.cov30(self.down(x20))
        x21 = self.cov21(torch.cat([x20, self.up(x30)], dim=1))
        x12 = self.cov12(torch.cat([x10, x11, self.up(x21)], dim=1))
        x03 = self.cov03(torch.cat([x00, x01, x02, self.up(x12)], dim=1))

        x40 = self.cov40(self.down(x30))
        x31 = self.cov31(torch.cat([x30, self.up(x40)], dim=1))
        x22 = self.cov22(torch.cat([x20, x21, self.up(x31)], dim=1))
        x13 = self.cov13(torch.cat([x10, x11, x12, self.up(x22)], dim=1))
        x04 = self.cov04(torch.cat([x00, x01, x02, x03, self.up(x13)], dim=1))

        final= self.final(x04)
        return torch.sigmoid(final)


if __name__ == '__main__':
    input = torch.randn(1, 3, 512, 512)
    model = UNet3Plus(3, 1)
    output = model(input)
    print(output)
