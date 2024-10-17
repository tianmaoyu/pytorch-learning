import torch

# 假设 bs=1, class_num=20
bs = 1
class_num = 20
original_feature_map = torch.randn(bs, 3*(5+class_num), 80, 80)

# 变形操作
# reshaped_feature_map = original_feature_map.view(bs, 3, 80, 80, 5+class_num)
reshaped_feature_map = original_feature_map.view(bs, 3, 5+class_num, 80, 80).permute(0, 1, 3, 4, 2).contiguous()

# 选择原始特征图中的一个位置
bs_index = 0
i = 10  # 任意网格坐标行
j = 10  # 任意网格坐标列
values_original = original_feature_map[bs_index, :, i, j]

# 选择变形后特征图中的相同位置
# values_reshaped = reshaped_feature_map[bs_index, :, i, j, :]
# reshape 会返回新的 张量，view 还是在原来的张量上共享内存
values_reshaped = reshaped_feature_map[bs_index, :, i, j, :].reshape(75,-1)

# 检查两个张量是否相同
print(torch.equal(values_original, values_reshaped))
