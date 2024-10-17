import torch

# 根据 三个不同的特征图，计算
# class_num=3; 5+3
# torch.Size([1, 24, 80, 80])
# torch.Size([1, 24, 40, 40])
# torch.Size([1, 24, 20, 20])


yv, xv = torch.meshgrid([torch.arange(20), torch.arange(20)])
grid= torch.stack((xv, yv), 2).view((1, 1, 20, 20, 2)).float()


# 根据特征图，生成网格 grid
grid_x, grid_y = torch.meshgrid([torch.arange(20), torch.arange(20)], indexing='ij')
grid_x = grid_x.float()
grid_y = grid_y.float()
grid_x = grid_x.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, height, width)
grid_y = grid_y.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, height, width)

# 示例：3 个锚框
anchors = torch.tensor([[10, 13], [16, 30], [33, 23]]).float()
pw = anchors[:, 0].view(1, 3, 1, 1)  # 锚框宽度
ph = anchors[:, 1].view(1, 3, 1, 1)  # 锚框高度

print(ph)
