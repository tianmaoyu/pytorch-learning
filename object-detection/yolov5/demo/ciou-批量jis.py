import torch
import torch.nn as nn

class CIoULoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, box1, box2, eps=1e-7):
        """
        计算两个边界框之间的 CIoU 值
        :param box1: 边界框1, [x1, y1, x2, y2]
        :param box2: 边界框2, [x1, y1, x2, y2]
        :return: CIoU 值
        """
        # 计算交集区域的坐标 (intersection)
        x1_inter = torch.max(box1[..., 0], box2[..., 0])  # 交集区域的左上角 x 坐标
        y1_inter = torch.max(box1[..., 1], box2[..., 1])  # 交集区域的左上角 y 坐标
        x2_inter = torch.min(box1[..., 2], box2[..., 2])  # 交集区域的右下角 x 坐标
        y2_inter = torch.min(box1[..., 3], box2[..., 3])  # 交集区域的右下角 y 坐标

        # 计算交集区域的宽度和高度
        inter_width = torch.clamp(x2_inter - x1_inter, min=0)
        inter_height = torch.clamp(y2_inter - y1_inter, min=0)

        # 交集面积
        inter_area = inter_width * inter_height

        # 计算每个框的面积
        box1_area = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
        box2_area = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])

        # 计算并集面积 (Union) eps 防止除0错误
        union_area = box1_area + box2_area - inter_area + eps

        # 计算 IoU
        iou = inter_area / union_area

        # 计算最小包围框的坐标 (enclosing box)
        x1_enclose = torch.min(box1[..., 0], box2[..., 0])
        y1_enclose = torch.min(box1[..., 1], box2[..., 1])
        x2_enclose = torch.max(box1[..., 2], box2[..., 2])
        y2_enclose = torch.max(box1[..., 3], box2[..., 3])

        # 计算最小包围框的对角线距离 (c^2)
        enclose_width = x2_enclose - x1_enclose
        enclose_height = y2_enclose - y1_enclose
        c2 = enclose_width ** 2 + enclose_height ** 2 + eps

        # 计算两个框的中心点 (center distance)
        center_box1_x = (box1[..., 0] + box1[..., 2]) / 2
        center_box1_y = (box1[..., 1] + box1[..., 3]) / 2
        center_box2_x = (box2[..., 0] + box2[..., 2]) / 2
        center_box2_y = (box2[..., 1] + box2[..., 3]) / 2
        d2 = (center_box1_x - center_box2_x) ** 2 + (center_box1_y - center_box2_y) ** 2

        # 计算 v 和 α (避免除以0的情况)
        w1 = box1[..., 2] - box1[..., 0]
        h1 = box1[..., 3] - box1[..., 1]
        w2 = box2[..., 2] - box2[..., 0]
        h2 = box2[..., 3] - box2[..., 1]

        v = (4 / (torch.pi ** 2)) * (torch.atan(w1 / (h1 + eps)) - torch.atan(w2 / (h2 + eps))) ** 2
        alpha = v / (1 - iou + v + eps)

        # 计算 CIoU
        ciou = iou - (d2 / c2) - alpha * v

        return 1 - ciou