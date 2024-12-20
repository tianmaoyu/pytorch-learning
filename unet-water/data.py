import os

import cv2
import torch
from torch import nn
from torchvision.transforms import functional
from torchvision.transforms.functional import pad
from functorch.dim import Tensor
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
import torch.nn
# from torchvision import  transforms
import shutil
import utils
# import albumentations as A
# from albumentations.pytorch import ToTensorV2

transform = transforms.Compose([
    transforms.ToTensor()
])

# # 定义图像和掩码的变换
# transform512 = A.Compose([
#     A.LongestMaxSize(max_size=512),  # 调整图片大小，保持长宽比不变
#     A.PadIfNeeded(min_height=512, min_width=512, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
#     # 填充图片到512x512
#     A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#     ToTensorV2()
# ])

transform512 = transforms.Compose([
    transforms.Resize(512),  # 保持长宽比，调整最短边为512像素
    transforms.CenterCrop(512),  # 从中心裁剪512x512的图片
    transforms.ToTensor()  # 转换成Tensor，方便后续处理
])

# 填充数据
def pad_16(image: Tensor) -> Tensor:
    height, width = functional.get_image_size(image)
    pad_height = (16 - height % 16) % 16
    pad_width = (16 - width % 16) % 16
    # 表示在上、下、左、右四个方向 mode：指定填充模式，可以是 “constant”、“reflect” 或 “replicate”；
    pad_image = nn.functional.pad(image, (0, pad_height, 0, pad_width), mode='replicate')
    return pad_image


class WaterDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.mask_path_list = []
        self.image_path_list = []

        self.mask_path = os.path.join(path, 'Annotations')
        self.image_path = os.path.join(path, 'JPEGImages')

        for dir_name, _, file_names in os.walk(self.mask_path):
            for file_name in file_names:
                mask_path = os.path.join(dir_name, file_name)
                image_path_jpg = mask_path.replace("Annotations", "JPEGImages").replace(".png", ".jpg")
                image_path_png = mask_path.replace("Annotations", "JPEGImages")

                if os.path.exists(image_path_jpg):
                    self.image_path_list.append(image_path_jpg)
                    self.mask_path_list.append(mask_path)
                elif os.path.exists(image_path_png):
                    self.image_path_list.append(image_path_png)
                    self.mask_path_list.append(mask_path)
                else:
                    print(f"找不到图片： {image_path_png}  ｛image_path_jpg｝")

    def __len__(self):
        return len(self.mask_path_list)

    def __getitem__(self, index) -> tuple[Tensor, Tensor]:
        image_path = self.image_path_list[index]
        image_mask_path = self.mask_path_list[index]

        try:
            image = Image.open(image_path).convert("RGB")
            mask_image = Image.open(image_mask_path).convert("L")

            image = transform(image)
            mask_image = transform(mask_image)
            # 获取张量的形状
            image_height, image_width = image.shape[-2:]
            mask_height, mask_width = mask_image.shape[-2:]

            return pad_16(image), pad_16(mask_image)
        except:
            print(f"数据错误: {image_path}")
            return  torch.randn(0,1,1,1),torch.randn(0,1,1,1)


class WaterDataset512(Dataset):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.mask_path_list = []
        self.image_path_list = []

        self.mask_path = os.path.join(path, 'Annotations')
        self.image_path = os.path.join(path, 'JPEGImages')

        for dir_name, _, file_names in os.walk(self.mask_path):
            for file_name in file_names:
                mask_path = os.path.join(dir_name, file_name)
                image_path_jpg = mask_path.replace("Annotations", "JPEGImages").replace(".png", ".jpg")
                image_path_png = mask_path.replace("Annotations", "JPEGImages")

                if os.path.exists(image_path_jpg):
                    self.image_path_list.append(image_path_jpg)
                    self.mask_path_list.append(mask_path)
                elif os.path.exists(image_path_png):
                    self.image_path_list.append(image_path_png)
                    self.mask_path_list.append(mask_path)
                else:
                    print(f"找不到图片： {image_path_png}  ｛image_path_jpg｝")

    def __len__(self):
        return len(self.mask_path_list)

    def __getitem__(self, index) -> tuple[Tensor, Tensor]:
        image_path = self.image_path_list[index]
        image_mask_path = self.mask_path_list[index]

        try:
            image = Image.open(image_path).convert("RGB")
            mask_image = Image.open(image_mask_path).convert("L")

            image = transform512(image)
            mask_image = transform512(mask_image)

            return image, mask_image
        except  Exception as e:
            print(f"数据错误: {image_path},{e}")
            return  torch.randn(0,1,1,1),torch.randn(0,1,1,1)



def copy_image(path: str):
    image_mask_path = os.path.join(path, 'SegmentationClass')
    image_name_list = os.listdir(image_mask_path)
    for image_name in image_name_list:
        image_path = os.path.join(path, "JPEGImages", image_name.replace('png', 'jpg'))
        destination_file_path = os.path.join(path, "seg-class-jpeg", image_name.replace('png', 'jpg'))
        # 复制文件
        shutil.copy(image_path, destination_file_path)


if __name__ == '__main__':

    pad_output = pad_16(torch.randn(1, 3, 252, 256))
    pad_output = pad_16(torch.randn(1, 3, 234, 234))
    pad_output = pad_16(torch.randn(1, 3, 456, 109))

    exit(1)

    writer = SummaryWriter("logs")
    water_dataset = WaterDataset(f"E:\语义分割\water_v2\water_v2")
    for i in range(10):
        image, mask_image = water_dataset[i]
        writer.add_image("image1", image, i)
        writer.add_image("mask_image1", mask_image, i)
    writer.close()
    exit(1)

    copy_image(f"E:\语义分割\VOCdevkit\VOC2012")
    exit(1)

    dateset = DemoDataset(f"E:\语义分割\VOCdevkit\VOC2012")
    # image,mask_image=dateset[0]
    writer = SummaryWriter("logs")

    for i in range(10):
        image, mask_image = dateset[i]
        writer.add_image("image", image, i)
        writer.add_image("mask_image", mask_image, i)

    writer.close()
    # writer.add_image("image",image,1 )
    # writer.add_image("mask_image",mask_image, 1)
    # writer.close()
    # pil_image =transforms.ToPILImage()
    # pil_image(image).show(title= image.image_name)
    # pil_image(mask_image).show(title= mask_image.image_name)
