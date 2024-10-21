import numpy as np
import torch




target_mask=np.zeros([1,3,80,80],dtype=bool)


target_mask[0,0,0,0]=True

target_box = np.zeros([1, 3, 80, 80, 85])

target_box[0,0,0,0,:4]=1,2,3,4
target_box[0,0,0,0,5+79]=1

filter=target_box[target_mask]
target_box= torch.tensor(target_box)

item = torch.zeros(85)
item[:4]=torch.tensor([1,1,1,2])
item[5:][79]=torch.tensor(1)



data=torch.tensor([[1.25000, 1.62500],
                   [2.00000, 3.75000],
                   [4.12500, 2.87500]])

index=torch.arange(3)
index2=torch.tensor([1,0,1])
out1= data[index,index2]
print(out1)
