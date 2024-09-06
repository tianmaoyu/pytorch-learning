import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np


# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        mask = Image.open(self.masks[idx])

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

    # 数据预处理和增强


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256))
])

# 加载数据集
train_images = ['path_to_train_image_1.jpg', 'path_to_train_image_2.jpg']  # 替换为实际的图像路径
train_masks = ['path_to_train_mask_1.png', 'path_to_train_mask_2.png']  # 替换为实际的掩码路径

train_dataset = CustomDataset(train_images, train_masks, transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# 创建DeepLabV3Plus模型
model = smp.DeepLabV3Plus(
    encoder_name="resnet34",  # 使用预训练的ResNet34作为编码器
    encoder_weights="imagenet",  # 使用在ImageNet上预训练的权重
    in_channels=3,  # 输入通道数（RGB图像）
    classes=1  # 输出通道数（分类数）
)

# 损失函数和优化器
criterion = torch.nn.BCEWithLogitsLoss()  # 二分类问题常用的损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader)}')

print("训练完成！")
