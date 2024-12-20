from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class BBoxCIoU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, box1, box2, eps=1e-7) -> Tensor:
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

        return ciou


#
class YoloV5Loss():
    def __init__(self):
        """
        :feature_map_num:  不同的特征层（不同分辨率）可以取 1，2，3 分别表示 :1: 80*80, 2:40*40; 3: 20*20
        """
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.bbox_ciou = BBoxCIoU()
        self.class_num = 80
        # layer上的 obj_loss weight
        self.layer_obj_loss_weight_list = [4.0, 1.0, 0.4]
        # 缩放比例
        self.layer_scale_list = [8, 16, 32]
        # layer 对应的 anchor 模板
        self.layer_anchors_list = [
            [[10, 13], [16, 30], [33, 23]],
            [[30., 61.], [62., 45.], [59., 119.]],
            [[116., 90.], [156., 198.], [373., 326.]]
        ]

    def __call__(self, predict_layer_list: List, label_list: Tensor):

        device = label_list.device
        batch_size = label_list.shape[0]
        box_loss = torch.zeros(1, device=device)
        cls_loss = torch.zeros(1, device=device)
        obj_loss = torch.zeros(1, device=device)

        for i, predict_layer in enumerate(predict_layer_list):
            layer_scale = self.layer_scale_list[i]
            layer_obj_loss_weight = self.layer_obj_loss_weight_list[i]
            layer_anchor = self.layer_anchors_list[i]

            layer_anchor = torch.tensor(layer_anchor).float() / layer_scale

            # 变形 [bs,3*(5+class_num),h,w] ->  [bs,3, (5+class_num),h,w] ->  [bs,3,h,w,(5+class_num)]
            bs, channel, height, width = predict_layer.shape
            predict_data = predict_layer.view(bs, 3, channel // 3, height, width).permute(0, 1, 3, 4, 2).contiguous()
            target, mask = self.build_target(label_list, layer_anchor, height, width, device)

            predict_obj = predict_data[..., 4]
            target_obj = torch.zeros(predict_obj.shape, device=device)

            # 是否匹配到正样本
            predict_positive = predict_data[mask]
            if predict_positive.shape[0] > 0:
                # 形状 [bs, 3, 80, 80,4]
                predict_box = self.build_predict_box(predict_data, layer_anchor)

                # 定位损失 :只计算正样本
                predict_box_positive = predict_box[mask]
                target_box_positive = target[mask][:, :4]
                ciou = self.bbox_ciou(predict_box_positive, target_box_positive)
                box_loss += (1.0 - ciou).mean()

                # 根据ciou计算而得
                target_obj[mask] = ciou.detach().clamp(0)

                # 分类损失 :只计算正样本
                predict_cls_positive = predict_data[mask][:, 5:]
                target_cls_positive = target[mask][:, 5:]
                cls_loss += self.bce_loss(predict_cls_positive, target_cls_positive)

            # 目标性损失 :正负样本都计算，并且每个layer 的权重也不一样
            obj_loss += self.bce_loss(predict_obj, target_obj) * layer_obj_loss_weight

        # 总损失 三个损失 权重系数：1; 0.5 ;0.005
        obj_loss = 1.0 * obj_loss
        cls_loss = 0.5 * cls_loss
        box_loss = 0.05 * box_loss
        loss = obj_loss + cls_loss + box_loss

        return loss * batch_size, torch.cat([box_loss, obj_loss, cls_loss, loss]).detach()

    def build_predict_box(self, predict_data, layer_anchor):
        # [bs,3,h,w,4]
        box = predict_data[..., :4].clone()

        xy = box[..., :2]
        box[..., :2] = 2 * torch.sigmoid(xy) - 0.5

        wh = box[..., 2:4]
        # 调整形状以匹配 [1,3,1,1,2]
        anchor_wh = layer_anchor.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        box[..., 2:4] = (2 * torch.sigmoid(wh)) ** 2 * anchor_wh

        return box

    def build_target(self, label_list, layer_anchor, height, width, device):
        """
        :param label_list:  [bs,target_num,6]  (image,label,x,y,w,h)
        :param layer_anchor:  [3,2]
        :param height:
        :param width:
        """
        batch, target_num = label_list.shape[:2]
        anchor_num = len(layer_anchor)
        # 初始化目标框张量，形状为 [bs, 3, 80, 80]
        target = np.zeros([batch, anchor_num, height, width, 85])
        mask = np.zeros([batch, anchor_num, height, width])

        # 分别循环批次，网格，anchor_num个层
        for b in range(batch):
            for i in range(target_num):
                label = label_list[b, i]
                class_index = label[1].long()
                true_x, true_y, true_w, true_h = label[2:6]
                # 计算对应网格的 x,y 坐标
                grid_x = int(true_x * width)
                grid_y = int(true_y * height)

                for k in range(anchor_num):
                    anchor_w, anchor_h = layer_anchor[k]

                    w_ratio = true_w / anchor_w
                    h_ratio = true_h / anchor_h

                    if (0.25 < w_ratio < 4) and (0.25 < h_ratio < 4):
                        mask[b, k, grid_y, grid_x] = True
                        # 注意：类别是 one-hot 编码
                        target[b, k, grid_y, grid_x, 5 + class_index] = 1
                        mod_x, mod_y = true_x - grid_x, true_y - grid_y
                        # 计算 x,y,w,h, 注意：x,y 是相对于该grid_x,grid_y坐标的偏移
                        target[b, k, grid_y, grid_x, :4] = mod_y, mod_x, true_w, true_h

                        # 匹配 网格 left,top,right,down 是否满足
                        # left
                        if mod_x < 0.5 and grid_x > 0:
                            target[b, k, grid_y, grid_x - 1, :4] = mod_y, mod_x + 1, true_w, true_h
                            target[b, k, grid_y, grid_x - 1, 5 + class_index] = 1
                            mask[b, k, grid_y, grid_x - 1] = True
                            # top
                        if mod_y < 0.5 and grid_y > 0:
                            target[b, k, grid_y - 1, grid_x, :4] = mod_y + 1, mod_x, true_w, true_h
                            target[b, k, grid_y - 1, grid_x, 5 + class_index] = 1
                            mask[b, k, grid_y - 1, grid_x] = True
                        # right
                        if mod_x > 0.5 and grid_x < width - 1:
                            target[b, k, grid_y, grid_x + 1, :4] = mod_y, mod_x - 1, true_w, true_h
                            target[b, k, grid_y, grid_x + 1, 5 + class_index] = 1
                            mask[b, k, grid_y, grid_x + 1] = True
                        # down
                        if mod_y > 0.5 and grid_y < height - 1:
                            target[b, k, grid_y + 1, grid_x, :4] = mod_y - 1, mod_x, true_w, true_h
                            target[b, k, grid_y + 1, grid_x, 5 + class_index] = 1
                            mask[b, k, grid_y + 1, grid_x] = True

        return torch.tensor(target, dtype=torch.float32,device=device), torch.tensor(mask, dtype=torch.bool,device=device)


if __name__ == '__main__':
    labels = torch.tensor([
        [2, 45, 0.479492, 0.688771, 0.955609, 0.5955],
        [2, 45, 0.736516, 0.247188, 0.498875, 0.476417],
        [2, 50, 0.637063, 0.732938, 0.494125, 0.510583],
        [2, 45, 0.339438, 0.418896, 0.678875, 0.7815],
        [2, 49, 0.646836, 0.132552, 0.118047, 0.0969375],
        [2, 49, 0.773148, 0.129802, 0.0907344, 0.0972292],
        [2, 49, 0.668297, 0.226906, 0.131281, 0.146896],
        [2, 49, 0.642859, 0.0792187, 0.148063, 0.148062]
    ])
    labels = labels.unsqueeze(0)
    layer1 = torch.rand([1, 3 * 85, 80, 80])
    layer2 = torch.rand([1, 3 * 85, 40, 40])
    layer3 = torch.rand([1, 3 * 85, 20, 20])
    layer_list = [layer1, layer2, layer3]

    loss = YoloV5Loss()

    result = loss(layer_list, labels)
    print(result)
