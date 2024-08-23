import torchvision
from torchvision import transforms

train_data=torchvision.datasets.CIFAR10(root="./dataset",train=True,download=True)
test_data =torchvision.datasets.CIFAR10(root="./dataset",train=False,download=True)

print(train_data,test_data)
image,label=train_data[0]
image.show()

transform_pipeline=transforms.Compose([
    transforms.ToTensor()
])
# 下载时转换
# train_data=torchvision.datasets.CIFAR10(root="./dataset",train=True,download=True,transform=transform_pipeline)

# 单个转换
image= transform_pipeline(image)
# 全部转换
transformed_data = []
# 遍历原始数据集，并应用转换
for image, label in train_data:
    transformed_image = transform_pipeline(image)
    transformed_data.append((transformed_image, label))


