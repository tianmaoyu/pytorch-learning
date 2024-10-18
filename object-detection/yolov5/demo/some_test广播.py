import torch

t = torch.randn(4, 4, 2)  # 形状: [3, 4, 2]
#
anchors = torch.randn([3,2])  # 形状: [3, 2]
#
anchors = torch.randn([2,2])  # 形状: [1, 2]

# 使用 unsqueeze 扩展 anchors 的维度
anchors_expanded = anchors.unsqueeze(1)  # 形状: [3, 1, 2]
# 逐元素相除
result = t / anchors_expanded
print(result.shape)
