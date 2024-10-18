# targets: Tensor, anchor_template: Tensor
import torch


def match_anchor():
    width = 80
    height = 80
    scale = 8
    # [6] image,label,x,y,w,h
    targets = torch.tensor([4, 22, 0.346211, 0.493259, 0.089422, 0.052118])
    # [3,2] w,h  缩小scale倍
    anchor_templates = torch.tensor([[10, 13], [16, 30], [33, 23]]).float() / scale
    # 还原到特征图
    targets[2] = targets[2] * width
    targets[3] = targets[3] * height
    targets[4] = targets[4] * width
    targets[5] = targets[5] * height

    #
    anchor_index_list = []
    # 匹配 高宽 比在： 0.25 -4 之间
    for index, anchor in enumerate(anchor_templates):
        w_ratio = targets[4] / anchor[0]
        h_ratio = targets[5] / anchor[1]
        if (0.25 < w_ratio < 4) and (0.25 < h_ratio < 4):
            anchor_index_list.append(index)

    t_x, t_y = targets[2:4]
    # 取小数点前面
    grid_x, grid_y = t_x.long(), t_y.long()
    # 取小数点后面
    mod_x, mod_y = t_x - grid_x, t_y - grid_y

    # 匹配相邻网格: left, top, right, down,
    grid_index_list = []
    grid_index_list.append([grid_x, grid_y])

    # left
    if mod_x < 0.5 and grid_x > 1.0:
        grid_index_list.append([grid_x - 1, grid_y])
    # top
    if mod_y < 0.5 and grid_y > 1.0:
        grid_index_list.append([grid_x, grid_y - 1])
    # right
    if mod_x > 0.5 and grid_x < width - 2:
        grid_index_list.append([grid_x + 1, grid_y])
    # down
    if mod_y > 0.5 and grid_y < height - 2:
        grid_index_list.append([grid_x, grid_y + 1])

    result = []
    for anchor_index in anchor_index_list:
        for grid_index in grid_index_list:
            result.append([anchor_index, grid_index])

    return result

# 匹配模板中的 anchor
def match_anchors():
    width = 80
    height = 80
    scale = 8
    # [6] image,label,x,y,w,h
    targets = torch.tensor([4, 22, 0.346211, 0.493259, 0.089422, 0.052118])
    # [3,2] w,h  缩小scale倍
    anchor_templates = torch.tensor([[10, 13], [16, 30], [33, 23]]).float() / scale
    # 还原到特征图
    targets[4] = targets[4] * width
    targets[5] = targets[5] * height

    #
    anchor_list = []
    # 匹配 高宽 比在： 0.25 -4 之间
    for index, anchor in enumerate(anchor_templates):
        w_ratio = targets[4] / anchor[0]
        h_ratio = targets[5] / anchor[1]
        if (0.25 < w_ratio < 4) and (0.25 < h_ratio < 4):
            anchor_list.append([index,anchor])

    return anchor_list

# 匹配周围的 grid  坐标
def match_grids():
    width = 80
    height = 80
    targets = torch.tensor([4, 22, 0.346211, 0.493259, 0.089422, 0.052118])
    # 还原到特征图
    targets[2] = targets[2] * width
    targets[3] = targets[3] * height


    t_x, t_y = targets[2:4]
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

