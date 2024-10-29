
import torch
import torch.optim as optim
import torch.nn as nn

model= nn.Sequential(nn.Linear(10, 10))

optimizer = optim.SGD(model.parameters(), lr=0.01,momentum=0.999, weight_decay=0.0005)
num_epochs=100
scheduler =optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs,eta_min=0.001)

for epoch in range(num_epochs):
    # 训练步骤
    # optimizer.step() and loss.backward() ...

    # 更新学习率
    scheduler.step()

    # 打印当前学习率
    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch + 1}/{num_epochs}, Learning Rate: {current_lr:.6f}")
