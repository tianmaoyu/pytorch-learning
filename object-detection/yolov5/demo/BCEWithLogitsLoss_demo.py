import torch
import torch.nn as nn

# 假设我们有一批次的 logits 和真实标签
logits = torch.tensor([100.0, 0.5, -1.0])  # 这可以是模型的原始输出
targets = torch.tensor([1.0, 0.0, 0])  # 真实标签

# 使用 BCEWithLogitsLoss,默认输出 loss 的均值
bce_loss= nn.BCEWithLogitsLoss()
loss = bce_loss(logits, targets)
print("BCEWithLogitsLoss:", loss.item())

# 手动计算过程：
sigmoid = torch.sigmoid(logits)  # 将 logits 转换为概率
# 添加小的 epsilon 防止 log(0)
epsilon = 1e-12  # 一个很小的值
manual_bce_loss = -(targets * torch.log(sigmoid) + (1 - targets) * torch.log(1 - sigmoid+epsilon))
manual_loss = manual_bce_loss.mean()  # 取均值
print("Manual BCE loss:", manual_loss.item())


#  类别损失 ---

# 假设有 3 个类别
num_classes = 3

# 创建 BCEWithLogitsLoss 实例,reduction="none" sum,
criterion = nn.BCEWithLogitsLoss(reduction="none")

# 模拟预测 logits，假设有 2 个样本，每个样本 3 个类别
logits = torch.tensor([[0.8, -1.2, 0.5],  # 第一个样本的 logits
                       [-0.5, 0.3, -0.1]])  # 第二个样本的 logits

# 模拟目标，假设目标是类别 1 和 0，one-hot 编码形式
# 第一个样本属于类别 0，第二个样本属于类别 1
targets = torch.tensor([[1.0, 0.0, 0.0],  # 第一个样本的目标
                        [0.0, 1.0, 0.0]])  # 第二个样本的目标

# 计算损失
loss = criterion(logits, targets)
print(loss.item())  # 输出损失值
