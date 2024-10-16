import glob
import os
import random

import torch
from PIL import Image
from functorch.dim import Tensor
from torch.utils.data import Dataset

transform=

class CocoDataset(Dataset):
    img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes

    def __init__(self, image_path, label_path, max_size=2000_000, ):
        """

        :param image_path: 图像目录
        :param label_path: lalel 目录
        :param max_size: 超过最大值进行缩放
        """
        super().__init__()
        self.max_size = max_size
        self.image_path = image_path
        self.label_path = label_path
        self.label_list = []
        self.image_list = []

        label_files = sorted(glob.glob(os.path.join(self.label_path, '*.txt')))

        # 根据 label 找到对应的图片
        for label_file in label_files:
            file_name = os.path.basename(label_file).split(".")[0]
            images = glob.glob(os.path.join(self.image_path, file_name + '.*'))
            images = [x for x in images if x.split('.')[-1].lower() in CocoDataset.img_formats]
            if len(images) < 1:
                continue

            self.label_list.append(label_file)
            self.image_list.append(images[0])

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index) -> tuple[Tensor, Tensor]:
        label_path = self.label_list[index]
        image_path = self.image_list[index]

        label = self._get_label(label_path)
        image = self._get_image(image_path)

        image = transform(image)
        label = transform(label)
        return image,label

    def _get_label(self, label_path) -> []:
        result = []
        with open(label_path, "r") as label_file:
            label_list = label_file.read().strip().splitlines()
            for line in label_list:
                list = [float(item) for item in line.split()]
                result.append(list)

        return result
    # ddd
    def _get_image(self, image_path) -> Image:
        try:
            image = Image.open(image_path).convert("RGB")
            # 图片太大，进行等比列缩放-内存不足
            width, height = image.size
            pixel_num = width * height
            if pixel_num > self.max_size:
                scale_factor = self.max_size / pixel_num
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                # 缩放图像,建议双线性插值
                image = image.resize((new_width, new_height), resample=Image.LANCZOS)

            return image
        except Exception as e:
            print(f"数据错误: {image_path}", e)
            return None


if __name__ == '__main__':
    image_path = "coco128/images/train2017"
    label_path = "coco128/labels/train2017"
    # CocoDataset(image_path, label_path)
    txt_file = "coco128/labels/train2017/000000000009.txt"
    with open(txt_file, "r") as label_file:
        label_list = label_file.read().strip().splitlines()
        point_list = []
        for line in label_list:
            text_list = line.split()
            # 只有一类，   # 删除第一个标签
            points = [float(item) for item in text_list]
            point_list.append(points)
        print(point_list)
