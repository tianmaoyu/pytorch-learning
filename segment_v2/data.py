import os
import random

import torch
from PIL import Image
from functorch.dim import Tensor
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.transforms import transforms, functional
import shutil
import utils

transform = transforms.Compose([
    transforms.ToTensor()
])


# 路径，和最大 图片像素
class WaterDataset(Dataset):
    def __init__(self, path, max_size=2000_000):
        super().__init__()
        self.path = path
        self.mask_path_list = []
        self.image_path_list = []
        self.max_size = max_size

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

            # 图片太大，进行等比列缩放-内存不足

            width, height = image.size
            pixel_num = width * height
            if pixel_num > self.max_size:
                scale_factor = self.max_size / pixel_num
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                # 缩放图像,建议双线性插值
                image = image.resize((new_width, new_height), resample=Image.LANCZOS)
                mask_image = mask_image.resize((new_width, new_height), resample=Image.LANCZOS)

            image = transform(image)
            mask_image = transform(mask_image)

            return utils.pad_16(image), utils.pad_16(mask_image)
        except Exception as e:
            print(f"数据错误: {image_path}", e)
            return torch.randn(0, 1, 1, 1), torch.randn(0, 1, 1, 1)


transform512 = transforms.Compose([
    transforms.Resize(512),  # 保持长宽比，调整最短边为512像素
    transforms.CenterCrop(512),  # 从中心裁剪512x512的图片
    transforms.ToTensor()  # 转换成Tensor，方便后续处理
])


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
            return torch.randn(0, 1, 1, 1), torch.randn(0, 1, 1, 1)


# 图像数据增强
class ExtendTransform:
    # 图像数据进行增强，
    def __init__(self, transform_type_list=["original", "h_flip", "v_flip", "rotate_90", "rotate_180", "rotate_270"]):
        self.transform_type_list = transform_type_list

    def __call__(self, image, mask):

        image, mask = functional.to_tensor(image), functional.to_tensor(mask)
        type = random.choice(self.transform_type_list)

        if type == "original":
            return image, mask
        if type == "h_flip":
            return functional.hflip(image), functional.hflip(mask)
        if type == "v_flip":
            return functional.vflip(image), functional.vflip(mask)
        if type == "rotate_90":
            return functional.rotate(image, 90, expand=True), functional.rotate(mask, 90, expand=True)
        if type == "rotate_180":
            return functional.rotate(image, 180), functional.rotate(mask, 180)
        if type == "rotate_270":
            return functional.rotate(image, 270, expand=True), functional.rotate(mask, 270, expand=True)


extend_transforms = ExtendTransform()


class ExtendWaterDataset(Dataset):
    def __init__(self, path, max_size=2000_000, extend_flag=True):
        super().__init__()
        # 数据是否进行增强
        self.extend_flag = extend_flag
        self.path = path
        self.mask_path_list = []
        self.image_path_list = []
        self.max_size = max_size

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

            # 图片太大，进行等比列缩放-内存不足
            width, height = image.size
            pixel_num = width * height
            if pixel_num > self.max_size:
                scale_factor = self.max_size / pixel_num
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                # 缩放图像,建议双线性插值
                image = image.resize((new_width, new_height), resample=Image.LANCZOS)
                mask_image = mask_image.resize((new_width, new_height), resample=Image.LANCZOS)

            # 是否增强
            if self.extend_flag:
                image, mask_image = extend_transforms(image, mask_image)
            else:
                image = transform(image)
                mask_image = transform(mask_image)

            return utils.pad_16(image), utils.pad_16(mask_image)
        except Exception as e:
            print(f"数据错误: {image_path}", e)
            return torch.randn(0, 1, 1, 1), torch.randn(0, 1, 1, 1)
