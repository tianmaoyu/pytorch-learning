import numpy as np
import  torch

loss= torch.nn.BCELoss(reduction="none")
pred_data_list=[[0.1,0.8,0.9],[0.2,0.7,0.8]]
pred_data= torch.tensor(pred_data_list,requires_grad=True)

traget_data_list=[[0.,0.,1.],[0.,0.,1.]]
traget_data=torch.tensor(traget_data_list)

loss_data=loss(pred_data,traget_data)

print(loss_data.detach().numpy())

#  自定义的 BCE  二值交叉熵
def custom_bce(c,o):
    return  -(o*np.log(c))+ (1-o)*np.log(1-c)

data =custom_bce(np.array(pred_data_list), np.array(traget_data_list))
print(data)