import torch

# 添加一个新维度，添加的新维度 为1 ：(3,2,1) 数据个数并不变
data1=torch.randn([3,2]).unsqueeze(2)
print(data1.shape)
#（挤压） 删除一个维度，默认1的，它的数据并不会变动
data2=torch.randn([3,2,1]).squeeze()
print(data2.shape)
# 添加维度另一个写法
data3= torch.randn([3,2])[:,:,None]
print(data3.shape)

# anchors 的形状 [3,2]
anchors = torch.tensor([[10, 13], [16, 30], [33, 23]])
# None在第 二维维度，变成  [3,1,2]
new_anchors = anchors[:, None]
#相同的效果
print(anchors.unsqueeze(1))