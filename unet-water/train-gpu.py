import torch
from torch.optim import *
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import functional
from data import WaterDataset
from net import UnetDemo
import logging

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = WaterDataset(f"D:\迅雷下载\water_v2\water_v2")
train_dataloader = DataLoader(dataset=dataset, batch_size=1)

model = UnetDemo(3, 1).to(device)
loss = nn.BCELoss().to(device)

optimizer = Adam(model.parameters(), 0.001)

for epoch in range(10):
    model.train()
    total_loss = 0
    for step, (images, mask_images) in enumerate(train_dataloader):

        images, mask_images = images.to(device), mask_images.to(device)

        height, width = functional.get_image_size(images)

        if height * width > 1000_000:
            continue

        model_result = model(images)

        loss_result = loss(model_result, mask_images)

        optimizer.zero_grad()
        loss_result.backward()
        optimizer.step()

        total_loss += loss_result.item()
        logger.info(f"epoch:{epoch} step: {step},loss: {loss_result.item()}")
        # if step % 10 == 0:
        #     # writer.add_scalar("lose-2", result_loss.item(), step)
        #     logger.info(f"epoch:{epoch} step: {step},loss: {loss_result.item()}")

    logger.info("---" * 20)
    logger.info(f"第 {epoch} 轮 total_loss:{total_loss}")
    torch.save(model, f"unet-{epoch}.pth")
