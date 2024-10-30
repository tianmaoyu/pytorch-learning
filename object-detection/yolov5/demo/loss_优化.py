import torch
from colorama import Fore
from torch import Tensor
from tqdm import tqdm


def build_target_1(predict_layer: Tensor, batch_label_list, layer_anchor):
    """
    :param predict_layer:
    :param batch_label_list:   (image, label, x, y, w, h)
    :param layer_anchor:  [3, 2]
    """
    device = predict_layer.device
    batch, channel, height, width = predict_layer.shape
    anchor_num = len(layer_anchor)

    # 初始化目标框张量，形状为 [bs, 3, 80, 80]
    target = torch.zeros([batch, anchor_num, height, width, 80 + 5], device=device)
    mask = torch.zeros([batch, anchor_num, height, width], device=device, dtype=torch.bool)

    # 获取图片索引列表
    batch_indices = batch_label_list[:, 0].long()
    class_indices = batch_label_list[:, 1].long()
    true_x = batch_label_list[:, 2] * width
    true_y = batch_label_list[:, 3] * height
    true_w = batch_label_list[:, 4] * width
    true_h = batch_label_list[:, 5] * height

    # 计算网格坐标
    grid_x = true_x.long()
    grid_y = true_y.long()
    mod_x = true_x - grid_x
    mod_y = true_y - grid_y

    # 将 layer_anchor 转换为张量
    layer_anchor = torch.tensor(layer_anchor, device=device, dtype=torch.float).view(1, anchor_num, 1, 1, 2)

    # 计算宽高比
    anchor_w = layer_anchor[:, :, :, :, 0]
    anchor_h = layer_anchor[:, :, :, :, 1]
    w_ratio = true_w.view(-1, 1, 1, 1) / anchor_w
    h_ratio = true_h.view(-1, 1, 1, 1) / anchor_h

    # 创建布尔掩码
    valid_mask = (0.25 < w_ratio) & (w_ratio < 4) & (0.25 < h_ratio) & (h_ratio < 4)

    # 更新目标张量和掩码
    for k in range(anchor_num):
        k_valid_mask = valid_mask[:, k, :, :, :]
        grid_x_k = grid_x[k_valid_mask]
        grid_y_k = grid_y[k_valid_mask]
        batch_indices_k = batch_indices[k_valid_mask]
        class_indices_k = class_indices[k_valid_mask]
        mod_x_k = mod_x[k_valid_mask]
        mod_y_k = mod_y[k_valid_mask]
        true_w_k = true_w[k_valid_mask]
        true_h_k = true_h[k_valid_mask]

        mask[batch_indices_k, k, grid_y_k, grid_x_k] = True
        target[batch_indices_k, k, grid_y_k, grid_x_k, 5 + class_indices_k] = 1
        target[batch_indices_k, k, grid_y_k, grid_x_k, :4] = torch.stack([mod_x_k, mod_y_k, true_w_k, true_h_k], dim=-1)

        # 处理相邻网格
        left_mask = (mod_x_k < 0.5) & (grid_x_k > 0)
        target[batch_indices_k[left_mask], k, grid_y_k[left_mask], grid_x_k[left_mask] - 1, :4] = torch.stack(
            [mod_x_k[left_mask] + 1, mod_y_k[left_mask], true_w_k[left_mask], true_h_k[left_mask]], dim=-1)
        target[batch_indices_k[left_mask], k, grid_y_k[left_mask], grid_x_k[left_mask] - 1, 5 + class_indices_k[
            left_mask]] = 1
        mask[batch_indices_k[left_mask], k, grid_y_k[left_mask], grid_x_k[left_mask] - 1] = True

        top_mask = (mod_y_k < 0.5) & (grid_y_k > 0)
        target[batch_indices_k[top_mask], k, grid_y_k[top_mask] - 1, grid_x_k[top_mask], :4] = torch.stack(
            [mod_x_k[top_mask], mod_y_k[top_mask] + 1, true_w_k[top_mask], true_h_k[top_mask]], dim=-1)
        target[
            batch_indices_k[top_mask], k, grid_y_k[top_mask] - 1, grid_x_k[top_mask], 5 + class_indices_k[top_mask]] = 1
        mask[batch_indices_k[top_mask], k, grid_y_k[top_mask] - 1, grid_x_k[top_mask]] = True

        right_mask = (mod_x_k > 0.5) & (grid_x_k < width - 1)
        target[batch_indices_k[right_mask], k, grid_y_k[right_mask], grid_x_k[right_mask] + 1, :4] = torch.stack(
            [mod_x_k[right_mask] - 1, mod_y_k[right_mask], true_w_k[right_mask], true_h_k[right_mask]], dim=-1)
        target[batch_indices_k[right_mask], k, grid_y_k[right_mask], grid_x_k[right_mask] + 1, 5 + class_indices_k[
            right_mask]] = 1
        mask[batch_indices_k[right_mask], k, grid_y_k[right_mask], grid_x_k[right_mask] + 1] = True

        down_mask = (mod_y_k > 0.5) & (grid_y_k < height - 1)
        target[batch_indices_k[down_mask], k, grid_y_k[down_mask] + 1, grid_x_k[down_mask], :4] = torch.stack(
            [mod_x_k[down_mask], mod_y_k[down_mask] - 1, true_w_k[down_mask], true_h_k[down_mask]], dim=-1)
        target[batch_indices_k[down_mask], k, grid_y_k[down_mask] + 1, grid_x_k[down_mask], 5 + class_indices_k[
            down_mask]] = 1
        mask[batch_indices_k[down_mask], k, grid_y_k[down_mask] + 1, grid_x_k[down_mask]] = True

    return target, mask


def build_target_2(predict_layer: Tensor, batch_label_list, layer_anchor):
    """
    :param predict_layer:
    :param batch_label_list:   (image,label,x,y,w,h)
    :param layer_anchor:  [3,2]
    """
    device = predict_layer.device
    batch, channel, height, width = predict_layer.shape

    anchor_num = len(layer_anchor)
    # 初始化目标框张量，形状为 [bs, 3, 80, 80]
    target = torch.zeros([batch, anchor_num, height, width, 80 + 5], device=device)
    mask = torch.zeros([batch, anchor_num, height, width], device=device, dtype=torch.bool)

    # 分别循环批次，网格，anchor_num个层
    for b in range(batch):

        # 筛选出 image_index =b 得图片数据
        label_list = batch_label_list[batch_label_list[:, 0] == b]

        for label in label_list:

            class_index = label[1].long()
            true_x, true_y, true_w, true_h = label[2:6]
            # 还原到比例： 比如 特征图 80*80 ，
            layer_x = true_x * width
            layer_y = true_y * height
            layer_w = true_w * width
            layer_h = true_h * height

            # 向下取正 得网格  x,y 坐标
            grid_x = layer_x.long()
            grid_y = layer_y.long()

            for k in range(anchor_num):
                anchor_w, anchor_h = layer_anchor[k]

                w_ratio = layer_w / anchor_w
                h_ratio = layer_h / anchor_h

                # 两个的比例在 0.25-4 之间
                if (0.25 < w_ratio < 4) and (0.25 < h_ratio < 4):
                    mask[b, k, grid_y, grid_x] = torch.tensor(True, device=device)
                    # 注意：类别是 one-hot 编码
                    target[b, k, grid_y, grid_x, 5 + class_index] = torch.tensor(1, device=device)
                    # 相对于网格坐标得偏移
                    mod_x, mod_y = layer_x - grid_x, layer_y - grid_y
                    # 计算 x,y,w,h, 注意：x,y 是相对于该grid_x,grid_y坐标的偏移
                    target[b, k, grid_y, grid_x, :4] = torch.tensor([mod_x, mod_y, layer_w, layer_h], device=device)

                    # 匹配 网格 left,top,right,down 是否满足
                    # left
                    if mod_x < 0.5 and grid_x > 0:
                        target[b, k, grid_y, grid_x - 1, :4] = torch.tensor([mod_x + 1, mod_y, layer_w, layer_h],
                                                                            device=device)
                        target[b, k, grid_y, grid_x - 1, 5 + class_index] = torch.tensor(1, device=device)
                        mask[b, k, grid_y, grid_x - 1] = torch.tensor(True, device=device)
                        # top
                    if mod_y < 0.5 and grid_y > 0:
                        target[b, k, grid_y - 1, grid_x, :4] = torch.tensor([mod_x, mod_y + 1, layer_w, layer_h],
                                                                            device=device)
                        target[b, k, grid_y - 1, grid_x, 5 + class_index] = torch.tensor(1, device=device)
                        mask[b, k, grid_y - 1, grid_x] = torch.tensor(True, device=device)
                    # right
                    if mod_x > 0.5 and grid_x < width - 1:
                        target[b, k, grid_y, grid_x + 1, :4] = torch.tensor([mod_x - 1, mod_y, layer_w, layer_h],
                                                                            device=device)
                        target[b, k, grid_y, grid_x + 1, 5 + class_index] = torch.tensor(1, device=device)
                        mask[b, k, grid_y, grid_x + 1] = torch.tensor(True, device=device)
                    # down
                    if mod_y > 0.5 and grid_y < height - 1:
                        target[b, k, grid_y + 1, grid_x, :4] = torch.tensor([mod_x, mod_y - 1, layer_w, layer_h],
                                                                            device=device)
                        target[b, k, grid_y + 1, grid_x, 5 + class_index] = torch.tensor(1, device=device)
                        mask[b, k, grid_y + 1, grid_x] = torch.tensor(True, device=device)

    return target, mask



if __name__ == '__main__':
    train_bar = tqdm(range(10000), total=10000, leave=True, postfix=Fore.WHITE)

    for i in train_bar:
       pass




