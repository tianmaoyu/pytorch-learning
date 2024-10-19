import torch


# 假设 data 的形状是 [1, 3, 20, 20, 85]
data = torch.rand([1, 3, 20, 20, 85])
# - `[1, 3]` 选择了第四维的索引 1 和 3。
# - `[0, 2, 4]` 选择了第五维的索引 0、2 和 4。
# 错误写法
# out = data[0, 0, 0, [1, 3], [0, 2, 4]]
# 实现1 :
out1 = data[0, 0, 0, [1, 3], :][:, [0, 2, 4]]  # 先选择前面的部分，然后再选择最后一维
out12 = data[[0], [0], [0], [1, 3], :][:, [0, 2, 4]]  # 先选择前面的部分，然后再选择最后一维

# 方式三； 好像不够灵活，比如我
dim_1_index = torch.tensor([0,0,0])
dim_2_index = torch.tensor([0,0,0])
dim_3_index = torch.tensor([0,1,2])
dim_4_index = torch.tensor([0,1,2])
out7=data[dim_1_index,dim_2_index,dim_3_index,dim_4_index]



dim_1_index = torch.tensor([0])
dim_2_index = torch.tensor([0])
dim_3_index = torch.tensor([0])
dim_4_index = torch.tensor([1, 3])
dim_5_index = torch.tensor([0, 2, 4])

# 手动使用  None 扩展维度
out2 = data[
    dim_1_index[:, None, None, None, None],
    dim_2_index[None, :, None, None, None],
    dim_3_index[None, None, :, None, None],
    dim_4_index[None, None, None, :, None],
    dim_5_index[None, None, None, None, :],
]
# 手动+ 自动广播扩展维度， 广播会先把 dim_5_index，dim_4_index,.... 扩展到 dim_1_index 一样；
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

# image,anchor,gridx,gridy,xywh+objectness+classes
data = torch.rand([1, 3, 20, 20, 5])
image_index = torch.tensor([0])
anchor_index = torch.tensor([1, 2])
gridx_index = torch.tensor([0, 1])
gridy_index = torch.tensor([0, 1])
data_index = torch.arange(0, 5)
# 我现在想的是，赛选出 image=0 全部,anchor=[1,2]... 的data; data 的形状 [1,2,2,2,85]
# 不是这种写法
# filter_data1 = data[image_index, anchor_index, gridx_index, gridy_index,data_index]
out = data[0, 0, 0, 0, 0]
out = data[[0], [0], [0], [0], [0]]
out = data[[0], [0], [0], [0], [0, 1, 2]]
out = data[[0], [0], [0], :1, [0, 1, 2]]  #
# out = data[[0], [0], [0], [0,2],[0,1,2]] # 错误写法
out = data[0, 0, 0, :2, :3]

filter_data2 = data[
    image_index[:, None, None, None, None],
    anchor_index[None, :, None, None, None],
    gridx_index[None, None, :, None, None],
    gridy_index[None, None, None, :, None],

]
print(filter_data2.shape)

filter_data3 = data[
    image_index[:, None, None, None, None],
    anchor_index[:, None, None, None],
    gridx_index[:, None, None],
    gridy_index[:, None],
    data_index
]
print(filter_data3.shape)

print(torch.equal(filter_data2, filter_data3))

data_2 = filter_data2[..., :]
data_3 = filter_data3[..., :]
print(torch.equal(data_2, data_3))
