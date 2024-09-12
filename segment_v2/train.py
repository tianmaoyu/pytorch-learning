import torch
from torch.optim import *
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import functional, transforms
from data import WaterDataset
import logging
from torchmetrics.segmentation import MeanIoU
import  unet

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

resize_transform = transforms.Compose([
    transforms.Resize(512),  # 保持长宽比，调整最短边为512像素
    transforms.CenterCrop(512),  # 从中心裁剪512x512的图片
])



# 合并 两个datasets
def merge_datasets(dataset1, dataset2):
    # 合并两个数据集的图像和掩码路径列表
    merged_image_paths = dataset1.image_path_list + dataset2.image_path_list
    merged_mask_paths = dataset1.mask_path_list + dataset2.mask_path_list

    # 使用合并后的路径列表创建一个新的数据集实例
    new_dataset = WaterDataset(f"E:\语义分割\water_v1")  # 传递None，因为我们不需要path属性
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

model =unet.Unet(3, 1).to(device)

loss = nn.BCELoss().to(device)

optimizer = Adam(model.parameters(), 0.001)


for epoch in range(20):
    model.train()
    total_loss = 0
    for step, (images, mask_images) in enumerate(train_dataloader):
        # todo 返回 None,或者 这句话写到下面 -会还是对图片进行 采集，会出现一个问题
        images, mask_images = images.to(device), mask_images.to(device)
        height, width = functional.get_image_size(images)

        if images.shape[1] > 3:
            continue
        if mask_images.shape[1] > 1:
            continue
        if images.shape[2] == 1:
            continue

        model_result = model(images)

        loss_result = loss(model_result, mask_images)

        optimizer.zero_grad()
        loss_result.backward()
        optimizer.step()

        # 指标
        total_loss += loss_result.item()
        mean_iou.update((model_result > 0.5).long(), mask_images.long())

        if step % 50 == 0:
            logger.info(f"epoch:{epoch} step: {step},loss: {loss_result.item()} iou:{mean_iou.compute().item()/50}")
            mean_iou.reset()

    # 验证
    model.eval()
    with torch.no_grad():
        for images, targets in val_dataloader:
            images, targets = images.to(device), targets.to(device)
            predictions = model(images)
            # 阈值化并转换为整数类型,  # 确保标签也是整数类型
            predictions = (predictions > 0.5).long()
            # 更新 MeanIoU
            mean_iou.update(predictions, targets.long())

        logger.info("---" * 20)
        logger.info(f"第 {epoch}  Mean IoU: {mean_iou.compute().item()/len(val_dataloader)} total_loss :{total_loss}")
        mean_iou.reset()
        torch.save(model, f"unet-{epoch}.pth")








