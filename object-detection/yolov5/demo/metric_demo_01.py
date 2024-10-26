import torch
from torchmetrics.detection  import MeanAveragePrecision
# 创建一个 torchmetrics 的 MAP 实例 内部使用了 coco 得工具包
metric = MeanAveragePrecision(box_format="xyxy")

# 模拟预测数据（preds）
# 假设我们有两张图片的预测结果
preds = [
    {
        "boxes": torch.tensor([[100, 150, 200, 250], [300, 400, 500, 600]]),  # 边界框的坐标
        "scores": torch.tensor([0.9, 0.75]),
        "labels": torch.tensor([1, 2])
    },
    {
        "boxes": torch.tensor([[50, 75, 125, 175]]),
        "scores": torch.tensor([0.85]),
        "labels": torch.tensor([2])
    }
]

# 模拟真实标签数据（gt）
gt = [
    {
        "boxes": torch.tensor([[110, 140, 200, 250], [310, 410, 510, 610]]),  # 边界框的坐标
        "labels": torch.tensor([1, 1])
    },
    {
        "boxes": torch.tensor([[50, 75, 125, 175]]),
        "labels": torch.tensor([2])
    }
]
# 更新指标，传入预测和真实数据
metric.update(preds, gt)
# 计算并输出 mAP 和其他评估指标
result = metric.compute()
print(result)
