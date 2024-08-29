import os
from functorch.dim import Tensor
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

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


if __name__ == '__main__':
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
