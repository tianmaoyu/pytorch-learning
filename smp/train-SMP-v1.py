import torch
from torch.optim import *
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import functional
from data import WaterDataset, WaterDataset512

import logging
import segmentation_models_pytorch as smp
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
    new_dataset = WaterDataset512(f"E:\语义分割\water_v1\water_v1")  # 传递None，因为我们不需要path属性
    new_dataset.image_path_list = merged_image_paths
    new_dataset.mask_path_list = merged_mask_paths

    return new_dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset1 = WaterDataset512(f"E:\语义分割\water_v1\water_v1")
dataset2 = WaterDataset512(f"E:\语义分割\water_v2\water_v2")

dataset = merge_datasets(dataset1, dataset2)

train_dataloader = DataLoader(dataset=dataset, batch_size=1)

# model = UnetDemoV1(3, 1).to(device)
model = smp.PSPNet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation="sigmoid"
).to(device)

smp.Unet(

)

loss = nn.BCELoss().to(device)
# loss = smp.losses.DiceLoss(mode="binary")
# smp.metrics.iou_score()
optimizer = Adam(model.parameters(), 0.001)

for epoch in range(10):
    model.train()
    total_loss = 0
    for step, (images, mask_images) in enumerate(train_dataloader):

        images, mask_images = images.to(device), mask_images.to(device)

        height, width = functional.get_image_size(images)

        if height==1 or width==1:
            continue

        # functional.resize()

        if height * width > 1000_000:
            continue

        if images.shape[1] > 3:
            continue
        if mask_images.shape[1] > 1:
            continue

        model_result = model(images)

        loss_result = loss(model_result, mask_images)

        optimizer.zero_grad()
        loss_result.backward()
        optimizer.step()

        total_loss += loss_result.item()
        if step % 10 == 0:
            logger.info(f"epoch:{epoch} step: {step},loss: {loss_result.item()}")

    logger.info("---" * 20)
    logger.info(f"第 {epoch} 轮 total_loss:{total_loss}")
    torch.save(model, f"unet-{epoch}.pth")
