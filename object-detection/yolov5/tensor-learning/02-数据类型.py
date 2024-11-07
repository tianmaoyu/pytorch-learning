import torch


# 默认的 torch.float32
random=torch.randn([2,3])
print(random.dtype)
# 抹掉小数，下取整数
random_int=random.int()
print(random_int.dtype)

# 不同设备上的数据不能操作
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
list=[1,2,3,4,5]
tensor_data= torch.tensor(list).to(device)
print(tensor_data.device)
data_list=torch.tensor([1,1,1,1,1])
print(data_list.device)
# 不同设备上的
list_data=tensor_data* data_list
print(list_data)

# mask