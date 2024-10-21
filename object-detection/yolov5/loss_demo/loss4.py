import torch
from torch import nn, Tensor


class YoloV5Loss:
    def __init__(self, feature_map_index):
        # 定义损失函数和超参数
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ciou_loss = CIoULoss()  # 假设这里有定义或实现的 CIoU 损失
        self.weight_obj = 1.0   # 权重: 目标置信度损失
        self.weight_cls = 0.5   # 权重: 分类损失
        self.weight_loc = 0.05  # 权重: 定位损失
        self.weight_obj_list = [4.0, 1.0, 0.4]  # 用于不同特征层的置信度损失权重

    def __call__(self, prediction: Tensor, labels: Tensor):
        # 获取预测张量的维度
        bs, channel, height, width = prediction.shape
        # 将预测结果进行 reshape，便于处理
        pred_data = prediction.view(bs, 3, channel // 3, height, width).permute(0, 1, 3, 4, 2).contiguous()

        # 分别获取预测的 x, y 坐标、宽高、置信度以及分类信息
        pred_x = pred_data[..., 0]
        pred_y = pred_data[..., 1]
        pred_w = pred_data[..., 2]
        pred_h = pred_data[..., 3]
        pred_obj = pred_data[..., 4]
        pred_cls = pred_data[..., 5:]

        # 获取 target 和 anchor/grid 信息
        target, other = self.build_targets(labels, pred_data.shape)  # 使用 build_targets 生成目标
        grid_x, grid_y, anchor_w, anchor_h = zip(*other)

        # 计算预测的 bbox 坐标
        bx, by, bw, bh = self.compute_grid_box(pred_x, pred_y, pred_w, pred_h, grid_x, grid_y, anchor_w, anchor_h)

        # 初始化目标置信度和分类标签的张量
        obj_target = torch.zeros_like(pred_obj)
        cls_target = torch.zeros_like(pred_cls)

        # 将匹配到的 target 信息填充到 obj_target 和 cls_target
        for anchor_idx in range(target.size(0)):
            for grid_idx in range(target.size(1)):
                gx, gy = grid_x[grid_idx], grid_y[grid_idx]
                obj_target[anchor_idx, :, gx, gy] = target[anchor_idx, grid_idx, 4]  # 置信度目标
                cls_target[anchor_idx, :, gx, gy] = target[anchor_idx, grid_idx, 5:]  # 分类目标

        # 计算定位损失 (CIoU)
        loc_loss = self.ciou_loss(torch.stack([bx, by, bw, bh], dim=-1), target[..., :4])

        # 计算目标置信度损失 (BCE)
        obj_loss = self.bce_loss(pred_obj, obj_target)

        # 计算分类损失 (BCE)
        cls_loss = self.bce_loss(pred_cls, cls_target)

        # 计算总损失
        loss = self.weight_loc * loc_loss + self.weight_obj * obj_loss + self.weight_cls * cls_loss

        return loss

    def compute_grid_box(self, pred_x, pred_y, pred_w, pred_h, grid_x, grid_y, anchor_w, anchor_h):
        """
        计算预测的 bounding box 参数
        :param pred_x: 预测的 x 坐标
        :param pred_y: 预测的 y 坐标
        :param pred_w: 预测的宽度
        :param pred_h: 预测的高度
        :param grid_x: 网格的 x 坐标
        :param grid_y: 网格的 y 坐标
        :param anchor_w: 锚框宽度
        :param anchor_h: 锚框高度
        :return: bx, by, bw, bh
        """
        # 中心坐标的偏移计算
        bx = 2 * torch.sigmoid(pred_x) - 0.5 + grid_x
        by = 2 * torch.sigmoid(pred_y) - 0.5 + grid_y
        # 宽高的计算
        bw = (2 * torch.sigmoid(pred_w)) ** 2 * anchor_w
        bh = (2 * torch.sigmoid(pred_h)) ** 2 * anchor_h
        return bx, by, bw, bh

    def build_predict_box(self, pred_data, layer_anchor):
        """
        :param pred_data:  [bs,3,80,80,85]
        :param layer_anchor: [3,2]
        :return: 预测框，形状为 [bs,3,80,80,4]
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
        # 形状 [bs, 4, 3, 80, 80] -> [bs,3,80,80,4]
        pred_box = pred_box.permute(0, 2, 3, 4, 1).contiguous()

        return pred_box


    def build_targets(self, labels, pred_shape):
        """
        构建目标匹配，用于计算损失
        :param labels: 输入的目标标签 (batch_size, num_targets, 6) - [batch_idx, class, x, y, w, h]
        :param pred_shape: 预测的形状，用于计算目标
        :return: target (匹配的目标标签), other (包含 grid 和 anchor 信息)
        """
        # 定义锚框以及网格的信息
        na = 3  # 锚框数量
        grid_size = pred_shape[-2:]  # 高度和宽度
        target_list = []
        other_info = []

        # 遍历 batch 和每个目标，进行匹配
        for label in labels:
            b, c, x, y, w, h = label.unbind(1)  # 将标签的6维拆解为分别为 batch_id, class, x, y, w, h
            gx, gy = x * grid_size[1], y * grid_size[0]  # 将中心点映射到网格坐标
            gw, gh = w * grid_size[1], h * grid_size[0]  # 将宽高映射到网格尺度

            # 匹配最合适的 anchor
            anchor_idx = self.match_anchor(gw, gh)
            target_list.append(torch.stack([gx, gy, gw, gh, anchor_idx], dim=-1))  # 生成目标信息
            other_info.append((gx, gy, gw, gh))

        target = torch.stack(target_list, dim=0)  # 堆叠成目标张量
        return target, other_info

    def match_anchor(self, gw, gh):
        """
        匹配最合适的锚框
        :param gw: 目标宽度
        :param gh: 目标高度
        :return: 锚框索引
        """
        # 根据宽高比匹配最合适的锚框 (这里假设已定义了 anchors)
        anchor_ratios = torch.tensor([[10, 13], [16, 30], [33, 23]])  # 假设为 YOLO 的 3 个锚框
        ratio = gw / gh
        anchor_idx = torch.argmin(torch.abs(ratio - anchor_ratios[:, 0] / anchor_ratios[:, 1]))
        return anchor_idx
