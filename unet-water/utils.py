from PIL import Image
import torch
from tensorboard.summary.v1 import image
from torch import Tensor, nn
from torchvision import io
import numpy as np
from torchvision.transforms import functional


def keep_image_size(path: str, size=(256, 256)) -> Image.Image:
    img = Image.open(path)
    max_size=max(img.size)
    mask = Image.new('RGB', (max_size, max_size), (0, 0, 0))
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    return mask

def keep_mask_image_size(path: str, size=(256, 256)) -> Image.Image:
    img = Image.open(path).convert("L")
    max_size=max(img.size)
    mask = Image.new('L', (max_size, max_size), 0)
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    return mask

def pad_16(image: Tensor) -> Tensor:
    width,height= functional.get_image_size(image)
    pad_height = (16 - height % 16) % 16
    pad_width = (16 - width % 16) % 16
    # 表示在左、右,上、下、四个方向 mode：指定填充模式，可以是 “constant”、“reflect” 或 “replicate”；
    pad_image = nn.functional.pad(image, (0, pad_width , 0, pad_height), mode='reflect')
    return pad_image

def show_image_memory_size(path: str):
    img = io.read_image(path)
    print(f"img shape:{img.shape}")
    print(f"data type:{img.dtype}")

    total_bytes = img.shape[0] * img.shape[1] * img.shape[2] * img.dtype.itemsize
    print(f"total bytes:{total_bytes},MB:{total_bytes/(1024*1024)}")


def sik_connection():
    # 假设有两个张量 dec4 和 enc4
    batch_size = 1
    channels_dec4 = 64
    channels_enc4 = 64
    height = 128
    width = 128

    dec4 = torch.randn(batch_size, channels_dec4, height, width)
    enc4 = torch.randn(batch_size, channels_enc4, height, width)

    print("Shape of dec4:", dec4.shape)
    print("Shape of enc4:", enc4.shape)

    # 在通道维度上拼接
    combined = torch.cat((dec4, enc4), dim=1)

    print("Shape of combined:", combined.shape)


if __name__ == '__main__':
    sik_connection()
    # show_image_memory_size("../data/test.jpg")
    # keep_image_size("../data/test.jpg")

