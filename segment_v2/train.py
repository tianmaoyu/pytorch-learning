import torch
from torch.optim import *
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import functional, transforms
from data import WaterDataset
import logging
from torchmetrics.segmentation import MeanIoU
from unet  import  Unet


# 训练停止条件，连续 多少次没有增长
class TrainStop:
    def __init__(self, count=3):
        self.count = count
        self.score_list = [0.0]
        self.best = 0.0
        self.trigger_count = 0

    def __call__(self, score: float) -> bool:
        self.score_list.append(score)
        total = sum(self.score_list[-self.count:])
        # 最佳分数： 最后几次平均分
        mean=total / self.count
        if  mean > self.best:
            self.best = mean

        # 分数没有超过之前，已经 count 次，就停止
        if self.best > score:
            self.trigger_count += 1
            if self.trigger_count > self.count+1:
                return True

        return False
# 日志
def config_logger(name="train"):
    # 设置日志的基本配置
    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler('app.log')
    file_handler.setLevel(logging.DEBUG)
    # 再创建一个handler，用于输出到控制台
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    # 定义handler的输出格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    # 给logger添加handler
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


# 合并 两个datasets
def merge_datasets(dataset_1, dataset_2):
    # 合并两个数据集的图像和掩码路径列表
    merged_image_paths = dataset_1.image_path_list + dataset_2.image_path_list
    merged_mask_paths = dataset_1.mask_path_list + dataset_2.mask_path_list

    # 使用合并后的路径列表创建一个新的数据集实例
    new_dataset = WaterDataset(dataset_1.path)
    new_dataset.image_path_list = merged_image_paths
    new_dataset.mask_path_list = merged_mask_paths

    return new_dataset

logger = config_logger()
train_stop = TrainStop(count=3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset1 = WaterDataset(f"E:\语义分割\water_v1\water_v1")
dataset2 = WaterDataset(f"E:\语义分割\water_v2\water_v2")
dataset = merge_datasets(dataset1, dataset2)

total_length = len(dataset)
var_length = int(total_length * 0.1)
train_dataset, val_dataset = random_split(dataset, [total_length - var_length, var_length])
train_dataloader = DataLoader(dataset=train_dataset, batch_size=1)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=1)


model = Unet(3, 1).to(device)
loss = nn.BCELoss().to(device)
optimizer = Adam(model.parameters(), 0.001)
mean_iou = MeanIoU(num_classes=2).to(device)

for epoch in range(20):
    # 训练
    model.train()
    total_loss = 0
    mean_iou.reset()
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
            logger.info(f"epoch:{epoch} step: {step},loss: {loss_result.item()} iou:{mean_iou.compute().item() / 50}")
            mean_iou.reset()

    # 验证
    model.eval()
    mean_iou.reset()
    var_mean_iou = 0
    with torch.no_grad():
        for images, targets in val_dataloader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            # 阈值化并转换为整数类型
            outputs = (outputs > 0.5).long()
            # 更新 MeanIoU- 内部是累加
            mean_iou.update(outputs, targets.long())

        logger.info("---" * 20)
        var_mean_iou = mean_iou.compute().item() / len(val_dataloader)
        logger.info(f"第epoch: {epoch}  Var Mean IoU: {var_mean_iou} total_loss :{total_loss}")

    torch.save(model, f"unet-{epoch}.pth")
    logger.info(f"模型已经保存：unet-{epoch}.pth")
    # 是否停止
    is_stop = train_stop(var_mean_iou)
    if is_stop:
        logger.info(f"停止训练: epoch:{epoch} 最佳iou {train_stop.best} , score_list:{train_stop.score_list}")
        break
