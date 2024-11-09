
import torch

# 假设 data 的形状是 [1, 3, 20, 20, 85]
data = torch.rand([1, 3, 20, 20, 85])

#普通
out = data[0, 0, 0, :2, :3]
# 假设  我在最后一维维维度上 取， 0，2，4 索引
 # 先选择前面的部分，然后再选择最后一维
out1 = data[0, 0, 0, [1, 3], :][:, [0, 2, 4]]

#取最后一维的 前三
out2=data[..., :3]

print(out1.shape)  # 输出结果的形状
print(out1)  # 查看结果  2,3

# 假设我现在要 取 指定的
# 花式索引
matrix = torch.tensor([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# 选取特定行
rows_to_select = [0, 2]  # 想要选取第1行和第3行
selected_rows = matrix[rows_to_select, :]

print(selected_rows)

# 其他
dim_1_index = torch.tensor([0])
dim_2_index = torch.tensor([0])
dim_3_index = torch.tensor([0])
dim_4_index = torch.tensor([1, 3])
dim_5_index = torch.tensor([0, 2, 4])

out2 = data[
    dim_1_index[:, None, None, None, None],
    dim_2_index[None, :, None, None, None],
    dim_3_index[None, None, :, None, None],
    dim_4_index[None, None, None, :, None],
    dim_5_index[None, None, None, None, :],
]

out3 = data[
    dim_1_index[:, None, None, None, None],
    dim_2_index[ :, None, None, None],
    dim_3_index[:, None, None],
    dim_4_index[:, None],
    dim_5_index,
]

print(out1.shape, out2.shape,out3.shape)
print(torch.equal(out2, out3))

print(torch.equal(out1, out2.squeeze()))

# bool 索引
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


# 生成一个三维张量
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
#这个会报错
#mask = torch.full([2,1], True, dtype=torch.bool)

# 使用布尔索引
result = z[mask]
print(result)