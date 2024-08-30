import torch
from torch import nn
from torchvision import transforms
from torch.nn import functional as F


class UnetDemo(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        def down_simple(channel):
           return nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=3, stride=2, padding=1, padding_mode="reflect", bias=False),
                nn.BatchNorm2d(channel),
                nn.LeakyReLU(inplace=True),
            )

        def block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode="reflect",bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode="reflect", bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(inplace=True),
            )

        self.encoder1=block(in_channel,64)
        self.encoder2=block(64,128)
        self.encoder3=block(128,256)
        self.encoder4=block(256,512)

        self.down1= down_simple(64)
        self.down2= down_simple(128)
        self.down3= down_simple(256)
        self.down4= down_simple(512)

        self.bottleneck = block(512, 1024)

        self.decoder1 = block(128, 64)
        self.decoder2 = block(256, 128)
        self.decoder3 = block(512, 256)
        self.decoder4 = block(1024, 512)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)

        self.final_conv = nn.Conv2d(64, out_channel, kernel_size=1)

    def forward(self, x):

        encode1=self.encoder1(x)
        down1=self.down1(encode1)

        encode2=self.encoder2(down1)
        down2=self.down2(encode2)

        encode3=self.encoder3(down2)
        down3=self.down3(encode3)

        encode4=self.encoder4(down3)
        down4=self.down4(encode4)

        bottleneck = self.bottleneck(down4)

        up4 = self.up4(bottleneck)
        cat4 = torch.cat((up4, encode4), dim=1)
        dec4 = self.decoder4(cat4)

        up3 = self.up3(dec4)
        cat3 = torch.cat((up3, encode3), dim=1)
        dec3 = self.decoder3(cat3)

        up2 = self.up2(dec3)
        cat2 = torch.cat((up2, encode2), dim=1)
        dec2 = self.decoder2(cat2)

        up1 = self.up1(dec2)
        cat1 = torch.cat((up1, encode1), dim=1)
        dec1 = self.decoder1(cat1)
        # 如果是多 分类可以使用 其他函数 如交叉熵
        return  torch.sigmoid(self.final_conv(dec1))





if __name__ == '__main__':
    in_data=torch.randn(1,3,256,256)
    model=UnetDemo(3,1)
    out_data= model(in_data)
    print(out_data.shape)



