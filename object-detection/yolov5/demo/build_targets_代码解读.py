import torch
# anchors 已经归一化
anchors = torch.tensor([[10, 13], [16, 30], [33, 23]]).float() / 8
anchors2 = torch.tensor([[30., 61.], [62., 45.], [59., 119.]]) / 16
anchors3 = torch.tensor([[116., 90.], [156., 198.], [373., 326.]]) / 32

prediction = torch.rand([1, 3, 20, 20, 85])
targets = torch.rand([1, 6])
anchor_num = 3
targets_num = 1

print(f"targets_num:{targets_num}")
gain = torch.ones(7)
ai = torch.arange(anchor_num).float().view(anchor_num, 1).repeat(1, targets_num)
targets = targets.repeat(anchor_num, 1, 1)
ai = ai.unsqueeze(2)
# 生成了[3,1,7]: 3 个 [image_id,label_id,x,y,w,h,index]
# 这里可以简单写
targets = torch.cat([targets, ai], dim=2)


w, h = prediction.shape[2:4]
gain[2:6] = torch.tensor([w, h, w, h])

#------根据 targets 匹配适合的 anchors--------
# 乘上对应的 xy,xy 长，宽
# targets 是归一画的，需要匹配 预测特征图尺度
t = targets * gain

# 没一个  组target 除上 对应的 anchor (三组)
# [3,targets_num,2] /[3,1,2]
#  targets_w/anchor_w,targets_h/anchor_h
r = t[:, :, 4:6] / anchors.unsqueeze(1)  # wh ratio

# 筛选 h,w 都满足 0.25-4 之间的数据
# r  shape:[3,1,2]  0.25-4 之间
mask1 = torch.max(r, 1 / r).max(2)[0]< 4
find1 =t[mask1]
# 判断 w,h 是否 0.25-4之间
mask2 = (r > 0.25) & (r < 4)
# w,h 必须都是 True: mask2[:, :, 0] 和 mask2[:, :, 1]
#[3,targets_num]
mask2 = mask2.all(dim=2)
# 这里会错误，如何改
find2=t[mask2]
print(find2)

#offsets
# x,y
gxy= find2[:,2:4]
gxi = gain[[2, 3]] - gxy  #
# gxy % 1. 计算小数部分
#  j,k,l,m 分别表示 左，上，右，下，； gxy>1. gxi > 1. 不是在图片的边界，
g = 0.5  # bias
off = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], device=targets.device).float() * g

j, k = ((gxy % 1. < g) & (gxy > 1.)).T
l, m = ((gxi % 1. < g) & (gxi > 1.)).T
# 第一个恒为 true 的 bool [5,1]
j = torch.stack((torch.ones_like(j), j, k, l, m))
# 重复 5 边，最后在 索引 扩维，在缩维度
find2 = find2.repeat((5, 1, 1))[j]
# 得出偏移 0.5 0 ;注意和上面 off 一一对应
offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
print(find2)

# image,class
# .long() 变成正数
b,c=find2[:,:2].long().T
gxy = find2[:, 2:4]  # grid xy
gwh = find2[:, 4:6]  # grid wh
# 变成整数取正  -offsets ,不是加 否者就对应不上了
gij = (gxy - offsets).long()
gi, gj = gij.T  # grid xy indices

# 取正数
a = find2[:, 6].long()  # anchor indices

tcls, tbox, indices, anch = [], [], [], []

gj_index= gj.clamp_(0, gain[3].long() - 1)
gi_index= gi.clamp_(0, gain[2].long() - 1)
indices.append((b, a,gj_index , gi_index))  # image, anchor, grid indices
tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
anch.append(anchors[a])  # anchors
tcls.append(c)  # class

print("xx")