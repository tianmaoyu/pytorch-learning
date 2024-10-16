import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class CIoULoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, box1, box2,eps=1e-7):
        """
        计算两个边界框之间的 CIoU 值
        :param box1: 边界框1, [x1, y1, x2, y2]
        :param box2: 边界框2, [x1, y1, x2, y2]
        :return: CIoU 值
        """
        # 计算交集区域的坐标 (intersection)
        x1_inter = max(box1[0], box2[0])  # 交集区域的左上角 x 坐标
        y1_inter = max(box1[1], box2[1])  # 交集区域的左上角 y 坐标
        x2_inter = min(box1[2], box2[2])  # 交集区域的右下角 x 坐标
        y2_inter = min(box1[3], box2[3])  # 交集区域的右下角 y 坐标

        # 计算交集区域的宽度和高度
        inter_width = max(0, x2_inter - x1_inter)
        inter_height = max(0, y2_inter - y1_inter)

        # 交集面积
        inter_area = inter_width * inter_height

        # 计算每个框的面积
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        # 计算并集面积 (Union) eps 防止0
        union_area = box1_area + box2_area - inter_area +eps

        # 计算 IoU
        iou = inter_area / union_area

        # 计算最小包围框的坐标 (enclosing box)
        x1_enclose = min(box1[0], box2[0])
        y1_enclose = min(box1[1], box2[1])
        x2_enclose = max(box1[2], box2[2])
        y2_enclose = max(box1[3], box2[3])

        # 计算最小包围框的对角线距离 (c^2)
        enclose_width = x2_enclose - x1_enclose
        enclose_height = y2_enclose - y1_enclose
        c2 = enclose_width ** 2 + enclose_height ** 2

        # 计算两个框的中心点 (center distance)
        center_box1_x = (box1[0] + box1[2]) / 2
        center_box1_y = (box1[1] + box1[3]) / 2
        center_box2_x = (box2[0] + box2[2]) / 2
        center_box2_y = (box2[1] + box2[3]) / 2
        d2 = (center_box1_x - center_box2_x) ** 2 + (center_box1_y - center_box2_y) ** 2

        # 计算 v 和 α
        w1 = box1[2] - box1[0]  # box1 的宽度
        h1 = box1[3] - box1[1]  # box1 的高度
        w2 = box2[2] - box2[0]  # box2 的宽度
        h2 = box2[3] - box2[1]  # box2 的高度

        v = (4 / np.pi ** 2) * (np.arctan(w1 / h1) - np.arctan(w2 / h2)) ** 2
        alpha = v / (1 - iou + v)

        # 计算 CIoU
        ciou = iou - (d2 / c2) - alpha * v
        # ciou_loss =1- ciou
        return 1-ciou


#
class YoloV5Loss():
    def __init__(self, feature_map_index):
        """
        :feature_map_num:  不同的特征层（不同分辨率）可以取 1，2，3 分别表示 :1: 80*80, 2:40*40; 3: 20*20

        """
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ciou_loss = CIoULoss()
        # 目标性损失的权重系数
        self.weight_obj = 1.0
        #  分类损失的权重系数
        self.weight_cls = 0.5
        # 定位损失的权重系数
        self.weight_loc = 0.05
        #不同特征层上不同权重系数
        self.weight_obj_list=[4.0, 1.0, 0.4]

    def __call__(self,  prediction: Tensor,target:Tensor):
        bs, channel, height, width = prediction.shape
        # 变形
        pred_data = prediction.view(bs, 3, channel // 3, height, width).permute(0, 1, 3, 4, 2).contiguous()
        #  获取预测的数据
        pred_x = pred_data[..., 0]
        pred_y = pred_data[..., 1]
        pred_w = pred_data[..., 2]
        pred_h = pred_data[..., 3]
        pred_obj = pred_data[..., 4]
        pred_cls = pred_data[..., 5:]

        # 根据特征图，生成网格 grid
        grid_x, grid_y = torch.meshgrid([torch.arange(height), torch.arange(width)], indexing='ij')
        grid_x = grid_x.float().to(prediction.device)
        grid_y = grid_y.float().to(prediction.device)
        grid_x = grid_x.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, height, width)
        grid_y = grid_y.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, height, width)

        # 示例：3 个锚框, 比如: 网格坐标(0,0) 位置对应 可以取出 三组 预测值，对应三个anchor box
        anchors = torch.tensor([[10, 13], [16, 30], [33, 23]], device=prediction.device).float()
        anchor_w = anchors[:, 0].view(1, 3, 1, 1)  # 锚框宽度
        anchor_h = anchors[:, 1].view(1, 3, 1, 1)  # 锚框高度

        bx, by, bw, bh = self.compute_grid_box(pred_x, pred_y, pred_w, pred_h, grid_x, grid_y, anchor_w, anchor_h)

        # 定位损失 (CIoU)
        loc_loss = self.ciou_loss(bx, by, bw, bh, target[..., :4])

        # 目标性损失 (BCE)
        obj_loss = self.bce_loss(pred_obj, target[..., 4])

        # 分类损失 (BCE)
        cls_loss = self.bce_loss(pred_cls, target[..., 5:])

        # 总损失
        loss = self.weight_loc * loc_loss + self.weight_obj * obj_loss + self.weight_cls * cls_loss

        return loss

    def compute_grid_box(self, pred_x, pred_y, pred_w, pred_h, grid_x, grid_y, anchor_w, anchor_h):
        # 中心坐标纠正
        bx = 2 * torch.sigmoid(pred_x) - 0.5 + grid_x
        by = 2 * torch.sigmoid(pred_y) - 0.5 + grid_y
        # 宽高纠正
        bw = (2 * torch.sigmoid(pred_w)) ** 2 * anchor_w
        bh = (2 * torch.sigmoid(pred_h)) ** 2 * anchor_h
        return bx, by, bw, bh





