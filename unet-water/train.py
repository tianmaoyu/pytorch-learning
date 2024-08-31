from  torch.optim  import *
from  torch import  nn
from torch.utils.data import DataLoader

from data import WaterDataset
from net import UnetDemo

dataset= WaterDataset(f"E:\语义分割\VOCdevkit\VOC2012")
train_dataloader=DataLoader(dataset=dataset,batch_size=12)

model = UnetDemo(3, 1)

loss=nn.BCELoss()
optimizer= Adam(model.parameters(),0.001)

for epoch in range(1000):
    model.train()
    total_loss=0
    step=0
    for images,mask_images in train_dataloader:

        model_result=model(images)

        loss_result=loss(model_result,mask_images)

        optimizer.zero_grad()
        loss_result.backward()
        optimizer.step()
        total_loss+=loss_result.item()
        if step % 100 == 0:
            # writer.add_scalar("lose-2", result_loss.item(), step)
            print(f" step {step},loss {loss_result.item()}")

    print("---"*20)
    print(f"第 {epoch} 轮 total_loss:{total_loss}")





