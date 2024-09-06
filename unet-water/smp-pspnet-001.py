import segmentation_models_pytorch as smp
from torch import Tensor, nn
from torchkeras import summary
import torch
from torchvision.transforms import functional
from  net_standard_v2 import UNetV2

# 填充数据
def pad_16(image: Tensor) -> Tensor:
    # 此处尺寸
    width, height = functional.get_image_size(image)
    pad_width = (16 - width % 16) % 16
    pad_height = (16 - height % 16) % 16
    # 表示在,左、右,上、下、四个方向 mode：指定填充模式，可以是 “constant”、“reflect” 或 “replicate”；
    pad_image = nn.functional.pad(image, (0, pad_width, 0, pad_height), mode='replicate')
    return pad_image


def demo_DeepLabV3Plus():
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation="sigmoid"
    )
    input = torch.randn(2, 3, 1000, 1000)
    input = pad_16(input)
    summary(model, input_data=input)
    output = model(input)
    print(output)

# demo_DeepLabV3Plus()

def demo_PSPNet_01():
    model = smp.PSPNet(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation="sigmoid"
    )
    input = torch.randn(1, 3, 473, 473)
    input = pad_16(input)
    # input=torch.randn(1,3,4000,3000)
    summary(model, input_data=input)
    output = model(input)
    print(output)


def demo_UNetV2():
    model = UNetV2(3, 1)
    input = torch.randn(1, 3, 1000, 1000)
    input = pad_16(input)
    # input=torch.randn(1,3,4000,3000)
    summary(model, input_data=input)
    output = model(input)
    print(output)

# demo_UNetV2()
demo_PSPNet_01()