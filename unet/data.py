import os

from fontTools.misc.cython import returns
from functorch.dim import Tensor
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import shutil
import utils

transform = transforms.Compose([
    transforms.ToTensor()
])


class DemoDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.image_mask_path = os.path.join(path, 'SegmentationClass')
        self.image_name_list = os.listdir(self.image_mask_path)

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, index) -> tuple[Tensor, Tensor]:
        image_name = self.image_name_list[index]
        image_mask_path = os.path.join(self.path, "SegmentationClass", image_name)
        image_path = os.path.join(self.path, "JPEGImages", image_name.replace('png', 'jpg'))

        image = utils.keep_image_size(image_path)
        mask_image = utils.keep_mask_image_size(image_mask_path)

        image = transform(image)
        # image.image_name=image_name.replace('png','jpg')
        mask_image = transform(mask_image)
        # mask_image.image_name=image_name

        return image, mask_image


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

                if os.path.exists(image_path_jpg) :
                    self.image_path_list.append(image_path_jpg)
                    self.mask_path_list.append(mask_path)
                elif os.path.exists(image_path_png) :
                    self.image_path_list.append(image_path_png)
                    self.mask_path_list.append(mask_path)
                else:
                    print(f"找不到图片： {image_path_png}  ｛image_path_jpg｝")



    def __len__(self):
        return len(self.mask_path_list)

    def __getitem__(self, index) -> tuple[Tensor, Tensor]:
        image_path = self.image_path_list[index]
        image_mask_path = self.mask_path_list[index]

        image = utils.keep_image_size(image_path)
        mask_image = utils.keep_mask_image_size(image_mask_path)

        image = transform(image)
        mask_image = transform(mask_image)

        return image, mask_image


def copy_image(path: str):
    image_mask_path = os.path.join(path, 'SegmentationClass')
    image_name_list = os.listdir(image_mask_path)
    for image_name in image_name_list:
        image_path = os.path.join(path, "JPEGImages", image_name.replace('png', 'jpg'))
        destination_file_path = os.path.join(path, "seg-class-jpeg", image_name.replace('png', 'jpg'))
        # 复制文件
        shutil.copy(image_path, destination_file_path)


if __name__ == '__main__':

    writer = SummaryWriter("logs")
    water_dataset= WaterDataset(f"E:\语义分割\water_v2\water_v2")
    for i in range(10):
        image, mask_image = water_dataset[i+2000]
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
