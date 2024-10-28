import torch


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

# 计算 IoU
def calculate_iou(boxA, boxB):
    """

    :param boxA:  [xyxy]
    :param boxB:  [xyxy]
    :return:
    """
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

# 初始化 TP, FP, FN 计数
tp, fp, fn = 0, 0, 0

# 遍历每张图片
for pred, gt in zip(preds, gt):
    pred_boxes = pred["boxes"]
    pred_scores = pred["scores"]
    pred_labels = pred["labels"]

    gt_boxes = gt["boxes"]
    gt_labels = gt["labels"]

    # 存储已匹配的真实框索引
    matched_gt = set()

    # 遍历每个预测框
    for i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
        matched = False
        for j, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):

            if j in matched_gt:
                continue  # 跳过已匹配的真实框
            if pred_label == gt_label:
                iou = calculate_iou(pred_box, gt_box)
                if iou >= iou_threshold:
                    matched_gt.add(j)  # 匹配成功，记录真实框索引
                    tp += 1
                    matched = True
                    break
        if not matched:
            fp += 1  # 没有匹配到的预测框记为 FP

    #漏掉的
    fn += len(gt_boxes) - len(matched_gt)

# 计算精确度、召回率和 F1 得分
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# 输出结果
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")
