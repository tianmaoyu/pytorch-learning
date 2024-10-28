import torch
from torchmetrics.detection  import MeanAveragePrecision


# 模拟预测数据（preds）
# 假设我们有两张图片的预测结果
pred_list = [
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
gt_list = [
    {
        "boxes": torch.tensor([[110, 140, 200, 250], [310, 410, 510, 610]]),  # 边界框的坐标
        "labels": torch.tensor([1, 1])
    },
    {
        "boxes": torch.tensor([[50, 75, 125, 175]]),
        "labels": torch.tensor([2])
    }
]
import torch
from collections import defaultdict


# 计算 IoU
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


# 设定 IoU 阈值
iou_threshold = 0.5

# 准备数据存储 AP 值
ap_per_class = defaultdict(list)

# 遍历每张图片
for pred, gt in zip(pred_list, gt_list):
    pred_boxes = pred["boxes"]
    pred_scores = pred["scores"]
    pred_labels = pred["labels"]

    gt_boxes = gt["boxes"]
    gt_labels = gt["labels"]

    # 针对每个类别计算 TP、FP，并保留对应的得分
    for c in torch.unique(pred_labels):
        c_pred_boxes = pred_boxes[pred_labels == c]
        c_pred_scores = pred_scores[pred_labels == c]
        c_gt_boxes = gt_boxes[gt_labels == c]

        # 按得分排序预测框
        sorted_indices = torch.argsort(c_pred_scores, descending=True)
        c_pred_boxes = c_pred_boxes[sorted_indices]
        c_pred_scores = c_pred_scores[sorted_indices]

        tp = torch.zeros(len(c_pred_boxes))
        fp = torch.zeros(len(c_pred_boxes))
        matched_gt = set()

        # 遍历预测框
        for i, pred_box in enumerate(c_pred_boxes):
            match_found = False
            for j, gt_box in enumerate(c_gt_boxes):
                if j in matched_gt:
                    continue
                iou = calculate_iou(pred_box, gt_box)
                if iou >= iou_threshold:
                    tp[i] = 1
                    matched_gt.add(j)
                    match_found = True
                    break
            if not match_found:
                fp[i] = 1

        # 计算累计 TP 和 FP
        tp_cumsum = torch.cumsum(tp, dim=0)
        fp_cumsum = torch.cumsum(fp, dim=0)

        # 计算精确度和召回率
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        recalls = tp_cumsum / len(c_gt_boxes)

        # 计算 AP
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        ap = torch.trapz(precisions, recalls).item()
        ap_per_class[c.item()].append(ap)

# 计算 mAP@0.5
mAP_50 = sum([sum(aps) / len(aps) for aps in ap_per_class.values()]) / len(ap_per_class)

# 输出结果
print(f"mAP@0.5: {mAP_50:.4f}")
