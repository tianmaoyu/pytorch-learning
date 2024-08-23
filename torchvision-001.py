import torchvision

train_data=torchvision.datasets.CIFAR10(root="./dataset",train=True,download=True)
test_data =torchvision.datasets.CIFAR10(root="./dataset",train=False,download=True)
print(train_data,test_data)

