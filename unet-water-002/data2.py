
import os
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
from datetime import datetime, timedelta
from glob import glob

# 假设 transform 和 pad_16 是你定义的图像转换和填充函数
transform = transforms.Compose([
    transforms.ToTensor()
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
    def __init__(self, path, max_time_diff=timedelta(minutes=1)):
        super().__init__()
        self.path = path
        self.mask_path_list = []
        self.image_path_list = []
        self.max_time_diff = max_time_diff

        self.mask_path = os.path.join(path, 'Annotations')
        self.image_path = os.path.join(path, 'JPEGImages')

        self._load_paths()

    def _load_paths(self):
        image_files = self._get_all_files(self.image_path)
        mask_files = self._get_all_files(self.mask_path)

        image_dict = {self._parse_datetime(f): f for f in image_files}

        for mask_file in mask_files:
            mask_name = os.path.basename(mask_file).rsplit('.', 1)[0]
            mask_time = self._parse_datetime(mask_file)

            # 尝试严格匹配
            matched_image = self._find_exact_match(mask_name, image_files)

            if not matched_image:
                # 如果没有严格匹配的结果，则尝试时间段匹配
                matched_image = self._find_time_range_match(mask_time, image_dict)

            if matched_image:
                self.mask_path_list.append(mask_file)
                self.image_path_list.append(matched_image)
            else:
                print(f"找不到对应图片: {mask_file}")

    def _get_all_files(self, directory):
        file_list = []
        for dirpath, _, filenames in os.walk(directory):
            for filename in filenames:
                file_list.append(os.path.join(dirpath, filename))
        return file_list

    def _parse_datetime(self, filepath):
        filename = os.path.basename(filepath)
        timestamp_str = filename.rsplit('.', 1)[0]
        try:
            return datetime.strptime(timestamp_str, '%Y-%m-%d-%H-%M-%S')
        except ValueError:
            return None

    def _find_exact_match(self, mask_name, image_files):
        for image_file in image_files:
            image_name = os.path.basename(image_file).rsplit('.', 1)[0]
            if mask_name == image_name:
                return image_file
        return None

    def _find_time_range_match(self, mask_time, image_dict):
        if mask_time is None:
            return None
        for image_time, image_file in image_dict.items():
            if abs((image_time - mask_time).total_seconds()) <= self.max_time_diff.total_seconds():
                return image_file
        return None

    def __len__(self):
        return len(self.mask_path_list)

    def __getitem__(self, index) -> tuple:
        image_path = self.image_path_list[index]
        image_mask_path = self.mask_path_list[index]
        try:
            image = Image.open(image_path).convert("RGB")
            mask_image = Image.open(image_mask_path).convert("L")
            image = transform(image)
            mask_image = transform(mask_image)
            return pad_16(image), pad_16(mask_image)
        except Exception as e:
            print(f"数据错误: {image_path}, 错误: {e}")
            return torch.randn(0, 1, 1, 1), torch.randn(0, 1, 1, 1)

if __name__ == '__main__':
    dataset=WaterDataset(f"E:\语义分割\water_v1\water_v")
    print(len(dataset))
