import torch
from torch.utils.data import Dataset
import os
from  typing import  Tuple
from PIL import Image


class myDataset(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(root_dir, label_dir)
        self.image_list = os.listdir(self.path)

    def __getitem__(self, index)->Tuple[str,str]:
        img = self.image_list[index]
        img_path=os.path.join(self.path,img)
        label_name = self.label_dir
        return img_path, label_name

    def __len__(self):
        return len(self.image_list)


x = torch.tensor([[1, 2, 3], [4, 5, 6]])
sum_rows = torch.sum(x, dim=0)  # 沿着行求和
sum_cols = torch.sum(x, dim=1)  # 沿着列求和

print(sum_rows)  # 输出: tensor([5, 7, 9])
print(sum_cols)  # 输出: tensor([ 6, 15])


sum_rows = torch.sum(x, dim=0,keepdim=True)  # 沿着行求和
sum_cols = torch.sum(x, dim=1,keepdim=True)  # 沿着列求和

print(sum_rows)  # 输出: tensor([5, 7, 9])
print(sum_cols)  # 输出: tensor([ 6, 15])
