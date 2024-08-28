


import os

import json
import numpy
import requests
import torch
import torchvision.datasets
from torch import nn
import torch.nn.functional as F
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models
from torchvision import transforms
# import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.pyplot as plt

os.environ['TORCH_HOME'] = "./dataset"
model = torchvision.models.vgg16(weights=models.VGG16_Weights.DEFAULT)
model_false = models.vgg16(weights=None)

print(model)

# 添加层
model.add_module("add_linear",nn.Linear(1000,10))
# 添加成
model.classifier.add_module("add_linear",nn.Linear(1000,10))
# 修改 或者添加 现有层  features ;classifier
model.features.add_module("7",nn.Linear(1000,10))
model.classifier[6]=nn.Linear(4096,10)
print(model)



file_path="./dataset/model-test.pth"
#保存方式一 ，模型结构+模型参数
torch.save(model,file_path)
model_test = torch.load(file_path)
print(model_test)

# 只保存模型参数; 加载后，用模型加载参数
torch.save(model.state_dict(),"state_dict.pth")
model.load_state_dict("state_dict.pth")
