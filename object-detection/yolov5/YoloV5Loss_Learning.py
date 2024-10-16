import torch
from torch import nn, Tensor


class YoloV5Loss_Learning():
    def __init__(self, weight_cls=1.0, weight_obj=1.0, weight_loc=1.0):
        """
        :param weight_cls:  分类损失的权重系数
        :param weight_obj: 目标性损失的权重系数
        :param weight_loc: 定位损失的权重系数
        """
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ciou_loss = CIoULoss()
        self.weight_cls = weight_cls
        self.weight_obj = weight_obj
        self.weight_loc = weight_loc


    def compute_grid_box(self, pred_x, pred_y, pred_w, pred_h, grid_x, grid_y, anchor_w, anchor_h):
        # 中心坐标纠正
        # bx = 2σ(x) - 0.5 + cx
        # by = 2σ(y) - 0.5 + cy
        bx = 2 * torch.sigmoid(pred_x) - 0.5 + grid_x
        by = 2 * torch.sigmoid(pred_y) - 0.5 + grid_y
        # 宽高纠正
        # bw = pw * 4 * σ(w)^2
        # bh = ph * 4 * σ(h)^2
        bw = (2 * torch.sigmoid(pred_w)) ** 2 *anchor_w
        bh =  (2 * torch.sigmoid(pred_h)) ** 2 *anchor_h
        return bx, by, bw, bh

    def __call__(self, prediction: Tensor,target:Tensor):
        # 假设prediction : 1, (5+3)*3,20,20
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
        # 根据grid  x,y 获取 anhor_box_list （3个）
        total_loss = 0  # 总损失
        # 分别循环批次，网格，3个层
        for b in range(bs):
            for i in range(height):
                for j in range(width):
                      for k in range(3):
                          x = pred_x[b, k, i, j]
                          y = pred_y[b, k, i, j]
                          w = pred_w[b, k, i, j]
                          h = pred_h[b, k, i, j]
                          obj = pred_obj[b, k, i, j]
                          cls = pred_cls[b, k, i, j]
                          # 网格x,y
                          g_x=grid_x[i, j]
                          g_y=grid_y[i, j]
                          # anchor box 中的 w,h
                          a_w=anchor_w[:, k]
                          a_h=anchor_h[:, k]
                          # 计算网格框 bx, by, bw, bh
                          bx, by, bw, bh = self.compute_grid_box(x, y, w, h, g_x, g_y, a_w, a_h)

                          # 获取目标框 target
                          target_box = target[b, k, i, j, :4]
                          target_obj = target[b, k, i, j, 4]
                          target_cls = target[b, k, i, j, 5:]

                          # 定位损失 (CIoU)
                          ciou = self.ciou_loss(bx, by, bw, bh, target_box)

                          # 目标性损失 (BCE)
                          obj_loss = self.bce_loss(obj, target_obj)

                          # 分类损失 (BCE)
                          cls_loss = self.bce_loss(cls, target_cls)

                          # 计算单个预测框的总损失
                          loss = self.weight_loc * ciou + self.weight_obj * obj_loss + self.weight_cls * cls_loss

                          # 累加损失
                          total_loss += loss

        return total_loss