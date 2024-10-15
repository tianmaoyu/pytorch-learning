import torch
import torch.nn as nn


class YOLOV1lastFc(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(1024 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 7 * 7 * 30)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        # 调整形状，一个形状 Flatten 展一维，最后使用 view 重塑 为张量
        x = x.view(-1, 7, 7, 30)
        return x


model = YOLOV1lastFc()
input_tensor = torch.randn(1, 1024, 7, 7)
output = model(input_tensor)
print(output)
vector_30d = output[0, 3, 4, :]  #提前3行，4列的网格
# 提取 深度方向的向量
all_vectors =output.view(-1,7*7,30)
print("所有网格的 30 维向量形状: ", all_vectors.shape)
