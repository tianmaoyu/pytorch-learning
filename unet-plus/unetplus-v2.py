import torch
import torch.nn as nn
import torch.nn.functional as F


class conv_block_nested(nn.Module):

    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)

        return output


# Nested Unet

class NestedUNet(nn.Module):
    """
    Implementation of this paper:
    https://arxiv.org/pdf/1807.10165.pdf
    """

    def __init__(self, in_ch=3, out_ch=1):
        super(NestedUNet, self).__init__()

        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = conv_block_nested(in_ch, 64, 64)
        self.conv1_0 = conv_block_nested(64, 128, 128)
        self.conv2_0 = conv_block_nested(128, 256, 256)
        self.conv3_0 = conv_block_nested(256, 512, 512)
        self.conv4_0 = conv_block_nested(512, 1024, 1024)

        self.conv0_1 = conv_block_nested(64 + 128, 64, 64)
        self.conv1_1 = conv_block_nested(128 + 256, 128, 128)
        self.conv2_1 = conv_block_nested(256 + 512, 256, 256)
        self.conv3_1 = conv_block_nested(512 + 1024, 512, 512)

        self.conv0_2 = conv_block_nested(64 * 2 + 128, 64, 64)
        self.conv1_2 = conv_block_nested(128 * 2 + 256, 128, 128)
        self.conv2_2 = conv_block_nested(256 * 2 + 512, 256, 256)

        self.conv0_3 = conv_block_nested(64 * 3 + 128, 64, 64)
        self.conv1_3 = conv_block_nested(128 * 3 + 256, 128, 128)

        self.conv0_4 = conv_block_nested(64 * 4 + 128, 64, 64)

        self.final = nn.Conv2d(64, out_ch, kernel_size=1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.down(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.down(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.down(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.down(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output


if __name__ == '__main__':

    for i in range(1, 3):
        print(i)
