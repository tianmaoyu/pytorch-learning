import torch


# 生成
y, x = torch.meshgrid(torch.arange(20), torch.arange(20))
#  [ny, nx, 2]  -> [1,1,ny,nx,2]
#https://blog.csdn.net/weixin_44201525/article/details/109769214
xy = torch.stack([x, y], 2).view(1, 1, 20, 20, 2)


