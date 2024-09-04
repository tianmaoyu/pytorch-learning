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
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.mask_path_list = []
        self.image_path_list = []

        self.mask_path = os.path.join(path, 'Annotations')
        self.image_path = os.path.join(path, 'JPEGImages')
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']

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

            return pad_16(image), pad_16(mask_image)
        except:
            print(f"数据错误: {image_path}")
            return  torch.randn(0,1,1,1),torch.randn(0,1,1,1)

def find_images(path,extensions=('jpg','jpeg','png','bmp','gif')):
    for dirpath, dirnames, filenames  in os.walk(path):
        for filename in filenames:
            if filename.lower().endswith(extensions):
                yield os.path.join(dirpath, filename)


class WaterDatasetV2(Dataset):
    def __init__(self, path, max_time_diff=timedelta(minutes=1)):
        super().__init__()
        self.path = path
        self.mask_path_list = []
        self.image_path_list = []
        self.max_time_diff = max_time_diff

        self.mask_path = os.path.join(path, r'Annotations\aberlour')
        self.image_path = os.path.join(path, r'JPEGImages\aberlour')

        self._load_paths()

    def _load_paths(self):
        image_files = self._get_all_files(self.image_path)
        mask_files = self._get_all_files(self.mask_path)

        image_dict = {self._parse_datetime(f): f for f in image_files}

        for mask_file in mask_files:
            mask_time = self._parse_datetime(mask_file)
            matched_image = self._find_matching_image(mask_time, image_dict)
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
        return datetime.strptime(timestamp_str, '%Y-%m-%d-%H-%M-%S')

    def _find_matching_image(self, mask_time, image_dict):
        for image_time, image_file in image_dict.items():
            if abs((image_time - mask_time).total_seconds()) <= self.max_time_diff.total_seconds():
                return image_file
        return None

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        # 实现获取item的方法
        pass

def rank_time():
    # 定义文件路径
    mask_paths = [
        'E:\\语义分割\\water_v2\\water_v2\\Annotations\\aberlour\\2019-08-03-08-00-10.png',
        'E:\\语义分割\\water_v2\\water_v2\\Annotations\\aberlour\\2019-08-03-08-03-10.png',
        'E:\\语义分割\\water_v2\\water_v2\\Annotations\\aberlour\\2019-08-03-10-30-10.png',
        'E:\\语义分割\\water_v2\\water_v2\\Annotations\\aberlour\\2019-08-03-13-00-10.png',
        'E:\\语义分割\\water_v2\\water_v2\\Annotations\\aberlour\\2019-08-03-15-30-10.png',
        'E:\\语义分割\\water_v2\\water_v2\\Annotations\\aberlour\\2019-08-03-17-57-10.png'
    ]

    jpeg_paths = [
        'E:\\语义分割\\water_v2\\water_v2\\JPEGImages\\aberlour\\2019-08-03-08-00-10.jpg',
        'E:\\语义分割\\water_v2\\water_v2\\JPEGImages\\aberlour\\2019-08-03-08-03-10.jpg',
        'E:\\语义分割\\water_v2\\water_v2\\JPEGImages\\aberlour\\2019-08-03-08-06-10.jpg',
        'E:\\语义分割\\water_v2\\water_v2\\JPEGImages\\aberlour\\2019-08-03-08-09-10.jpg',
        'E:\\语义分割\\water_v2\\water_v2\\JPEGImages\\aberlour\\2019-08-03-08-12-10.jpg',
        'E:\\语义分割\\water_v2\\water_v2\\JPEGImages\\aberlour\\2019-08-03-08-15-10.jpg'
        # 等等，添加所有的JPEG路径
    ]

    # 从路径中提取时间戳
    def extract_timestamp(file_path):
        file_name = os.path.basename(file_path)
        timestamp_str = file_name.split('.')[0]
        return datetime.strptime(timestamp_str, '%Y-%m-%d-%H-%M-%S')

        # 将路径和时间戳绑定在一起

    mask_timestamps = [(path, extract_timestamp(path)) for path in mask_paths]
    jpeg_timestamps = [(path, extract_timestamp(path)) for path in jpeg_paths]

    # 按时间戳排序
    mask_timestamps.sort(key=lambda x: x[1])
    jpeg_timestamps.sort(key=lambda x: x[1])

    # 进行匹配
    result = {}
    for mask_path, mask_time in mask_timestamps:
        matched_jpegs = [jpeg_path for jpeg_path, jpeg_time in jpeg_timestamps if jpeg_time <= mask_time]
        result[mask_path] = matched_jpegs

        # 输出结果
    for mask_path, matched_jpegs in result.items():
        print(f"Mask: {mask_path}")
        for jpeg_path in matched_jpegs:
            print(f"  Matched JPEG: {jpeg_path}")

def rank_time2():
    import os
    from datetime import datetime

    # 定义文件路径
    mask_paths = [
        'E:\\语义分割\\water_v2\\water_v2\\Annotations\\aberlour\\2019-08-03-08-00-10.png',
        'E:\\语义分割\\water_v2\\water_v2\\Annotations\\aberlour\\2019-08-03-08-03-10.png',
        'E:\\语义分割\\water_v2\\water_v2\\Annotations\\aberlour\\2019-08-03-10-30-10.png',
        'E:\\语义分割\\water_v2\\water_v2\\Annotations\\aberlour\\2019-08-03-13-00-10.png',
        'E:\\语义分割\\water_v2\\water_v2\\Annotations\\aberlour\\2019-08-03-15-30-10.png',
        'E:\\语义分割\\water_v2\\water_v2\\Annotations\\aberlour\\2019-08-03-17-57-10.png'
    ]

    jpeg_paths = [
        'E:\\语义分割\\water_v2\\water_v2\\JPEGImages\\aberlour\\2019-08-03-08-00-10.jpg',
        'E:\\语义分割\\water_v2\\water_v2\\JPEGImages\\aberlour\\2019-08-03-08-03-10.jpg',
        'E:\\语义分割\\water_v2\\water_v2\\JPEGImages\\aberlour\\2019-08-03-08-06-10.jpg',
        'E:\\语义分割\\water_v2\\water_v2\\JPEGImages\\aberlour\\2019-08-03-08-09-10.jpg',
        'E:\\语义分割\\water_v2\\water_v2\\JPEGImages\\aberlour\\2019-08-03-08-12-10.jpg',
        'E:\\语义分割\\water_v2\\water_v2\\JPEGImages\\aberlour\\2019-08-03-08-15-10.jpg'
        # 等等，添加所有的JPEG路径
    ]

    # 从路径中提取时间戳
    def extract_timestamp(file_path):
        file_name = os.path.basename(file_path)
        timestamp_str = file_name.split('.')[0]
        return datetime.strptime(timestamp_str, '%Y-%m-%d-%H-%M-%S')

        # 将路径和时间戳绑定在一起

    mask_timestamps = [(path, extract_timestamp(path)) for path in mask_paths]
    jpeg_timestamps = [(path, extract_timestamp(path)) for path in jpeg_paths]

    # 按时间戳排序
    mask_timestamps.sort(key=lambda x: x[1])
    jpeg_timestamps.sort(key=lambda x: x[1])

    # 进行匹配
    result = {}
    mask_index = 0
    mask_count = len(mask_timestamps)

    for jpeg_path, jpeg_time in jpeg_timestamps:
        while mask_index < mask_count and mask_timestamps[mask_index][1] < jpeg_time:
            mask_index += 1
        if mask_index < mask_count:
            mask_path = mask_timestamps[mask_index][0]
            if mask_path not in result:
                result[mask_path] = []
            result[mask_path].append(jpeg_path)

            # 输出结果
    for mask_path, matched_jpegs in result.items():
        print(f"Mask: {mask_path}")
        for jpeg_path in matched_jpegs:
            print(f"  Matched JPEG: {jpeg_path}")


if __name__ == '__main__':
    rank_time2()
    data= WaterDatasetV2(r"E:\语义分割\water_v2\water_v2")
    exit(1)

    path= r"E:\语义分割\water_v2\water_v2\JPEGImages\ADE20K\ADE_val_00000114.png"
    # basename=os.path.basename(path)
    #
    # ss= os.path.splitext(path)
    #
    # mask_path= path.replace("JPEGImages","Annotations")
    #
    v2_file_num=0
    for root, dirs, files in os.walk(f"E:\语义分割\water_v2\water_v2"):
        v2_file_num+=len(files)
        print(f"文件夹:{root},files:{len(files)}")
        for file in files:
            print(os.path.join(root,file))

    v1_file_num=0
    for root, dirs, files in os.walk(f"E:\语义分割\water_v1\water_v1"):
        v1_file_num+=len(files)
        print(f"文件夹:{root},files:{len(files)}")
        for file in files:
            print(os.path.join(root,file))

    print(f"v1 total file num:{v1_file_num}")
    print(f"v2 total file num:{v2_file_num}")


