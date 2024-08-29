
from  torch.optim  import *
from  torch import  nn
from torch.utils.data import DataLoader

from unet.data import DemoDataset
from unet.net import UnetDemo


dataset= DemoDataset(f"E:\语义分割\VOCdevkit\VOC2012")
train_dataloader=DataLoader(dataset=dataset,batch_size=4)

model = UnetDemo(3, 1)

loss=nn.BCELoss()
optimizer= Adam(model.parameters(),0.01)


for epoch in range(1):
    model.train()
    step=0
    for images,mask_images in train_dataloader:

        model_result=model(images)

        loss_result=loss(model_result,mask_images)

        optimizer.zero_grad()
        loss_result.backward()
        optimizer.step()

        if step % 100 == 0:
            # writer.add_scalar("lose-2", result_loss.item(), step)
            print(f" step {step},loss {loss_result.item()}")







