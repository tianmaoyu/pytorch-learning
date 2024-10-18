import torch

# 生成一个形状为 [3, 4] 的张量
x = torch.tensor([[10, 20, 30, 40],
                  [50, 60, 70, 80],
                  [90, 100, 110, 120]])

# 生成一个布尔张量，形状与 x 相同
mask = torch.tensor([[True, False, True, False],
                     [False, True, False, True],
                     [True, False, True, False]])

# 使用布尔索引 一维张量
result = x[mask]
print(result)




# 生成一个三维张量 [3,2,2]
y = torch.tensor([[[1, 2], [3, 4]],
                  [[5, 6], [7, 8]],
                  [[9, 10], [11, 12]]])

# 生成一个布尔张量，只有第 0 维有布尔值，0 维上过滤得到2个
mask = torch.tensor([True, False, True])

# 得到 [2，2，2]
result = y[mask]
print(result)


# 生成一个形状为 [2, 3, 4] 的张量
z = torch.tensor([[[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]],

                  [[13, 14, 15, 16],
                   [17, 18, 19, 20],
                   [21, 22, 23, 24]]])
# [2,3]
mask = torch.full([2,3], True, dtype=torch.bool)

# 使用布尔索引
result = z[mask]
print(result)