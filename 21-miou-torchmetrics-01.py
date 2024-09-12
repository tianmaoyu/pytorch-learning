import os
import torch
import matplotlib.pyplot as plt
import numpy
from torchmetrics.classification import BinaryJaccardIndex,JaccardIndex
from torchmetrics.segmentation import MeanIoU


def meaniou_001():
    # 假设我们有两个批次的预测和真实标签
    preds = torch.tensor([[[0, 1], [1, 0]], [[1, 0], [0, 1]]])
    target = torch.tensor([[[0, 1], [1, 0]], [[0, 1], [1, 0]]])
    # 创建 MeanIoU 实例
    mean_iou = MeanIoU(num_classes=2)
    # 更新状态- 里面会进行累加
    mean_iou.update(preds, target)
    mean_iou.update(preds, target)
    mean_iou.update(preds, target)
    # 计算平均 IoU
    iou = mean_iou.compute()
    print(f"Mean IoU: {iou}")


def meaniou():
    mean_iou=  MeanIoU(num_classes=2)
    values = []

    target = torch.randint(0, 2, (1, 10, 10), dtype=torch.long)
    # 生成并计算 10 次随机数据的 IoU 值
    for _ in range(10):
        for _ in range(10):
            # 生成随机预测和标签
            outputs = torch.randint(0, 2, (1, 10, 10), dtype=torch.long)
            # outputs=target
            # preds = torch.argmax(outputs, dim=1)
            mean_iou.update(outputs, target)

        values.append(mean_iou.compute()/10)
        mean_iou.reset()
    # 绘制 IoU 值的变化曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(0, len(values)), numpy.array(values)/100, marker='o')
    plt.ylim(0, 1)
    plt.title('mean iou/10')
    plt.xlabel('epoch')
    plt.ylabel('IoU Value')
    plt.grid(True)
    plt.show()

meaniou_001()



def jaccardindex():
    # 创建 Binary Jaccard Index 指标实例
    metric = BinaryJaccardIndex()
    # 初始化一个列表来存储每次迭代的 IoU 值
    values = []

    # 生成并计算 10 次随机数据的 IoU 值
    for _ in range(10):
        # 生成随机预测和标签
        preds = torch.randn(10)  # 预测概率
        target = torch.randint(low=0, high=2, size=(10,))  # 实际标签
        # 将预测概率转换为二进制标签
        preds_binary = (preds > 0.5).float()
        # 更新并获取 IoU 值
        value = metric(preds_binary, target)
        values.append(value.item())

    # 绘制 IoU 值的变化曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 11), values, marker='o')
    plt.title('IoU Values Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('IoU Value')
    plt.grid(True)
    plt.show()
