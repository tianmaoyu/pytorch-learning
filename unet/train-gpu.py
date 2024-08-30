import torch
from  torch.optim  import *
from  torch import  nn
from torch.utils.data import DataLoader

from unet.data import DemoDataset
from unet.net import UnetDemo

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset= DemoDataset(f"E:\语义分割\VOCdevkit\VOC2012")
train_dataloader=DataLoader(dataset=dataset,batch_size=12)

model = UnetDemo(3, 1).to(device)
loss=nn.BCELoss().to(device)

optimizer= Adam(model.parameters(),0.001)

for epoch in range(10):
    model.train()
    total_loss=0
    for step, (images, mask_images) in enumerate(train_dataloader):

        images, mask_images = images.to(device), mask_images.to(device)

        model_result=model(images)

        loss_result=loss(model_result,mask_images)

        optimizer.zero_grad()
        loss_result.backward()
        optimizer.step()

        total_loss+=loss_result.item()
        if step % 10 == 0:
            # writer.add_scalar("lose-2", result_loss.item(), step)
            print(f"epoch:{epoch} step: {step},loss: {loss_result.item()}")

    print("---"*20)
    print(f"第 {epoch} 轮 total_loss:{total_loss}")
    torch.save(model,f"unet-{epoch}.pth")





