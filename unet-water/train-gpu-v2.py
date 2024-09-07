import torch
from torch.optim import *
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import functional
from data import WaterDataset
from net_standard_v1 import UnetDemoV1
import logging
from torchmetrics.segmentation import MeanIoU

# 设置日志的基本配置
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# 创建一个handler，用于写入日志文件
fh = logging.FileHandler('app.log')
fh.setLevel(logging.DEBUG)
# 再创建一个handler，用于输出到控制台
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# 定义handler的输出格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# 给logger添加handler
logger.addHandler(fh)
logger.addHandler(ch)


# 合并 两个datasets
def merge_datasets(dataset1, dataset2):
    # 合并两个数据集的图像和掩码路径列表
    merged_image_paths = dataset1.image_path_list + dataset2.image_path_list
    merged_mask_paths = dataset1.mask_path_list + dataset2.mask_path_list

    # 使用合并后的路径列表创建一个新的数据集实例
    new_dataset = WaterDataset(f"D:\迅雷下载\water_v1\water_v1")  # 传递None，因为我们不需要path属性
    new_dataset.image_path_list = merged_image_paths
    new_dataset.mask_path_list = merged_mask_paths

    return new_dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset1 = WaterDataset(f"E:\语义分割\water_v1\water_v1")
dataset2 = WaterDataset(f"E:\语义分割\water_v2\water_v2")

dataset = merge_datasets(dataset1, dataset2)
total_length = len(dataset)
var_length = int(total_length * 0.1)
train_dataset, val_dataset = random_split(dataset, [total_length - var_length, var_length])
train_dataloader = DataLoader(dataset=train_dataset, batch_size=1)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=1)

mean_iou = MeanIoU(num_classes=2).to(device)
model = UnetDemoV1(3, 1).to(device)
loss = nn.BCELoss().to(device)

optimizer = Adam(model.parameters(), 0.001)
mean_iou = MeanIoU(num_classes=2).to(device)
for epoch in range(10):
    # 验证
    model.eval()
    with torch.no_grad():
        for images, targets in val_dataloader:
            images, targets = images.to(device), targets.to(device)
            predictions = model(images)
            # 阈值化并转换为整数类型,  # 确保标签也是整数类型
            predictions = (predictions > 0.5).long()
            targets = targets.long()
            # 更新 MeanIoU
            mean_iou.update(predictions, targets)

        logger.info("---" * 20)
        logger.info(f"第 {epoch} 轮验证 mIoU: {mean_iou.compute().item():.4f}")  # 计算mIoU并记录
        mean_iou.reset()



class Accuracy(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean_iou = MeanIoU(num_classes=2)
        self.count=0

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        predictions = (preds > 0.5).long()
        targets = targets.long()
        self.mean_iou.update(predictions, targets)
        self.count+=1
        return self.mean_iou.compute()/ self.count

    def compute(self):
         return self.mean_iou.compute()/ self.count

    def reset(self):
        self.mean_iou.reset()
