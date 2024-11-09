import torch
import torchvision

# 定义候选框，形状为 [N, 4]，每个框 [x1, y1, x2, y2]
boxes = torch.tensor([[10, 10, 20, 20],  # 第一个框
                      [12, 12, 22, 22],  # 第二个框，和第一个框重叠
                      [100, 100, 150, 150],  # 第三个框
                      [110, 110, 160, 160],  # 第四个框，和第三个框重叠
                      [10, 10, 20, 20],  # 第五个框，属于另一类别
                      [12, 12, 22, 22]]).to(torch.float)  # 第六个框，属于另一类别

# 定义每个候选框的置信度得分
scores = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.95, 0.85]).to(torch.float)

# 定义每个候选框的类别标签
labels = torch.tensor([0, 0, 1, 1, 2, 2]).to(torch.float)  # 0 类别, 1 类别, 2 类别

# IoU 阈值
iou_threshold = 0.45

# 调用 batched_nms, 索引会根据分数排序
indices = torchvision.ops.batched_nms(boxes, scores, labels, iou_threshold)
# 数据也更根据索引位置 排序位置一一对应
filter_boxes= boxes[indices]
# 输出保留的框索引
print(filter_boxes)




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

