import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


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
        union_area = box1_area + box2_area - inter_area + eps

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
        return 1 - ciou


#
class YoloV5Loss():
    def __init__(self):
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

    def __call__(self, pred_layer_list: list[Tensor], label: Tensor):
        loc_loss = torch.zeros(1, device=label.device)
        cls_loss = torch.zeros(1, device=label.device)
        obj_loss = torch.zeros(1, device=label.device)

        for i, pred_layer in enumerate(pred_layer_list):
            layer_scale = self.layer_scale_list[i]
            layer_obj_loss_weight = self.layer_obj_loss_weight_list[i]
            layer_anchor = self.layer_anchors_list[i]

            #
            layer_anchor = torch.tensor(layer_anchor).float() / layer_scale

            bs, channel, height, width = pred_layer.shape

            # 变形
            pred_data = pred_layer.view(bs, 3, channel // 3, height, width).permute(0, 1, 3, 4, 2).contiguous()

            # 形状 [bs, 4, 3, 80, 80]
            pred_box = self.build_pred_box(pred_data, layer_anchor)

            pred_obj = pred_data[..., 4]
            pred_cls = pred_data[..., 5:]

            target_box = self.build_target_box(pred_data, label, layer_anchor, height, width)
            target_obj = self.build_target_obj(pred_data, label, layer_anchor)
            target_cls = self.build_target_cls(pred_data, label, layer_anchor)

            # 定位损失 (CIoU) 只计算正样本
            loc_loss += self.ciou_loss(pred_box, target_box)

            # 目标性损失 (BCE) 正负样本都计算
            obj_loss += self.bce_loss(pred_obj, target_obj)

            # 分类损失 (BCE)  只计算正样本
            cls_loss += self.bce_loss(pred_cls, target_cls)

        # 总损失
        loss = self.weight_loc * loc_loss + self.weight_obj * obj_loss + self.weight_cls * cls_loss

        return loss

    def build_pred_box(self, pred_data, layer_anchor):
        """
        :param pred_data:  [bs,3,80,80,85]
        :param layer_anchors: [3,2]
        :return: 预测框，形状为 [4, bs, 3, 80, 80]
        """
        # [3]
        anchor_w = layer_anchor[..., 0]
        anchor_h = layer_anchor[..., 1]
        # [bs, 3, 80, 80]
        pred_x = pred_data[..., 1]
        pred_y = pred_data[..., 2]
        pred_w = pred_data[..., 3]
        pred_h = pred_data[..., 4]

        # 中心坐标纠正
        x = 2 * torch.sigmoid(pred_x) - 0.5
        y = 2 * torch.sigmoid(pred_y) - 0.5
        # [1,3,1,1]
        anchor_w = anchor_w.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        anchor_h = anchor_h.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        # 宽高纠正
        w = (2 * torch.sigmoid(pred_w)) ** 2 * anchor_w
        h = (2 * torch.sigmoid(pred_h)) ** 2 * anchor_h

        # 形状 [bs, 4, 3, 80, 80]
        pred_box = torch.stack([x, y, w, h], dim=1)

        return pred_box

    def build_target_box(self, pred_data, labels, layer_anchor, height, width):
        """
        :param pred_data: [bs,3,80,80,85]
        :param label:  [bs,target_num,6]  (image,label,x,y,w,h)
        :param layer_anchor:  [3,2]
        :param height:
        :param width:
        """
        bs,target_num = labels.shape[:2]
        anchor_num= len(layer_anchor)
        # 初始化目标框张量，形状为 [bs, 3, 80, 80]
        target_box = torch.zeros((bs, anchor_num, height, width, 4), device=pred_data.device)  # 4是 [x, y, w, h]

        # 分别循环批次，网格，anchor_num个层
        for b in range(bs):
            for i in range(target_num):
                label = labels[b, i]

                true_x, true_y, true_w, true_h = label[2:6]
                # 计算对应网格的 x,y 坐标
                grid_x = int(true_x * width)
                grid_y = int(true_y * height)

                for k in range(anchor_num):
                    anchor_w, anchor_h = layer_anchor[k]

                    w_ratio = true_w / anchor_w
                    h_ratio = true_h / anchor_h

                    if (0.25 < w_ratio < 4) and (0.25 < h_ratio < 4):
                        # 填充目标框数据
                        target_box[b, k, grid_y, grid_x] = torch.tensor([true_x, true_y, true_w, true_h], device=pred_data.device)

                        # 匹配 left,top,right,down 是否满足
                        # 取小数点后面
                        mod_x, mod_y = true_x - grid_x, true_y - grid_y
                        # left
                        if mod_x < 0.5 and grid_x > 0:
                            target_box[b, k, grid_y, grid_x- 1] = torch.tensor([true_x, true_y, true_w, true_h], device=pred_data.device)
                        # top
                        if mod_y < 0.5 and grid_y > 0:
                            target_box[b, k, grid_y-1, grid_x] = torch.tensor([true_x, true_y, true_w, true_h], device=pred_data.device)
                        # right
                        if mod_x > 0.5 and grid_x < width - 1:
                            target_box[b, k, grid_y, grid_x+1] = torch.tensor([true_x, true_y, true_w, true_h], device=pred_data.device)
                        # down
                        if mod_y > 0.5 and grid_y < height - 1:
                            target_box[b, k, grid_y+1, grid_x] = torch.tensor([true_x, true_y, true_w, true_h], device=pred_data.device)




        # 返回目标框信息，形状为 [bs, anchor_num, height, width, 4]
        return target_box
















def build_target_obj(self, pred_data, label, layer_anchors):
    pass


def build_target_cls(self, pred_data, label, layer_anchors):
    pass


def compute_pred_box(self, pred_x, pred_y, pred_w, pred_h, grid_x, grid_y, anchor_w, anchor_h):
    # 中心坐标纠正
    bx = 2 * torch.sigmoid(pred_x) - 0.5 + grid_x
    by = 2 * torch.sigmoid(pred_y) - 0.5 + grid_y
    # 宽高纠正
    bw = (2 * torch.sigmoid(pred_w)) ** 2 * anchor_w
    bh = (2 * torch.sigmoid(pred_h)) ** 2 * anchor_h
    return bx, by, bw, bh


# 匹配模板中的 anchor
def match_anchors(labels: Tensor, anchor_templates: Tensor, size=(80, 80)):
    width, height = size
    # 还原到特征图
    labels[4] = labels[4] * width
    labels[5] = labels[5] * height

    anchor_list = []
    # 匹配 高宽 比在： 0.25 -4 之间
    for index, anchor in enumerate(anchor_templates):
        w_ratio = labels[4] / anchor[0]
        h_ratio = labels[5] / anchor[1]
        if (0.25 < w_ratio < 4) and (0.25 < h_ratio < 4):
            anchor_list.append([index, anchor])

    return anchor_list


# 匹配周围的 grid  坐标
def match_grids(labels: Tensor, size=(80, 80)):
    width, height = size
    # 还原到特征图
    labels[2] = labels[2] * width
    labels[3] = labels[3] * height

    t_x, t_y = labels[2:4]
    # 取小数点前面
    grid_x, grid_y = t_x.long(), t_y.long()
    # 取小数点后面
    mod_x, mod_y = t_x - grid_x, t_y - grid_y

    # 匹配相邻网格: left, top, right, down,
    grid_list = []
    grid_list.append([grid_x, grid_y])
    # left
    if mod_x < 0.5 and grid_x > 1.0:
        grid_list.append([grid_x - 1, grid_y])
    # top
    if mod_y < 0.5 and grid_y > 1.0:
        grid_list.append([grid_x, grid_y - 1])
    # right
    if mod_x > 0.5 and grid_x < width - 2:
        grid_list.append([grid_x + 1, grid_y])
    # down
    if mod_y > 0.5 and grid_y < height - 2:
        grid_list.append([grid_x, grid_y + 1])

    return grid_list


def build_targets():
    width = 80
    height = 80
    scale = 8
    labels = torch.tensor([4, 22, 0.346211, 0.493259, 0.089422, 0.052118])
    # [6] image,label,x,y,w,h
    # [3,2] w,h  缩小scale倍
    anchor_templates = torch.tensor([[10, 13], [16, 30], [33, 23]]).float() / scale
    anchor_list = match_anchors(labels, anchor_templates, size=(width, height))
    grid_list = match_grids(labels, size=(width, height))

    # 这里要组装一个 tragets,和 other
    targets = torch.zeros((len(anchor_list), len(grid_list), 6))  # 6维: [x, y, w, h, obj_conf, cls]
    other = []

    # 填充 target
    for anchor_idx, anchor in anchor_list:
        for grid_idx, (grid_x, grid_y) in enumerate(grid_list):
            # 计算与网格和 anchor 的偏移
            tx = labels[2] * width - grid_x
            ty = labels[3] * height - grid_y
            tw = torch.log((labels[4] * width) / anchor[0])
            th = torch.log((labels[5] * height) / anchor[1])

            # 填入到 target 中
            targets[anchor_idx, grid_idx, 0] = tx
            targets[anchor_idx, grid_idx, 1] = ty
            targets[anchor_idx, grid_idx, 2] = tw
            targets[anchor_idx, grid_idx, 3] = th
            targets[anchor_idx, grid_idx, 4] = 1  # object confidence
            targets[anchor_idx, grid_idx, 5] = labels[1]  # 分类标签 (cls)

            # 把对应的 grid 和 anchor 信息保存在 other 中
            other.append([grid_x, grid_y, anchor[0], anchor[1]])

    return targets, other


if __name__ == '__main__':
    labels = torch.tensor([[4, 22, 0.346211, 0.493259, 0.089422, 0.052118]])
    labels = labels.unsqueeze(0)
    layer1 = torch.rand([1, 3 * 85, 80, 80])
    layer2 = torch.rand([1, 3 * 85, 40, 40])
    layer3 = torch.rand([1, 3 * 85, 20, 20])
    layer_list = [layer1, layer2, layer3]

    loss = YoloV5Loss()

    loss(layer_list, labels)
