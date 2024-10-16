import glob
import os

from PIL import Image
from functorch.dim import Tensor
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage,functional

import utils


class CocoDataset(Dataset):
    img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes

    def __init__(self, image_path, label_path):
        super().__init__()

        self.image_path = image_path
        self.label_path = label_path
        self.label_list = []
        self.image_list = []

        if not os.path.exists(label_path):
            print(f"路径 {label_path} 不存在")

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

    def __getitem__(self, index) -> tuple[Tensor, list[list[float]]]:
        label_path = self.label_list[index]
        image_path = self.image_list[index]

        labels = self._get_label(label_path)
        image = Image.open(image_path).convert("RGB")

        # 填充32 倍数，和缩放 640
        image,labels=utils.letterbox(image,labels)

        image= functional.to_tensor(image)

        return image, labels

    def _get_label(self, label_path) -> []:
        result = []
        with open(label_path, "r") as label_file:
            label_list = label_file.read().strip().splitlines()
            for line in label_list:
                list = [float(item) for item in line.split()]
                result.append(list)

        return result



if __name__ == '__main__':
    image_path = "coco128/images/train2017"
    label_path = "coco128/labels/train2017"
    dataset= CocoDataset(image_path, label_path)
    image,labels=dataset[0]

    print(labels)

    to_pil_image = ToPILImage()

    plt.figure(dpi=300)

    img= to_pil_image(image)
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    img_draw=utils.draw_rectangle(img,labels)
    plt.imshow(img_draw)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

