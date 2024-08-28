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

# 设置环境变量
os.environ['TORCH_HOME'] = "./dataset"
model = torchvision.models.vgg16(weights=models.VGG16_Weights.DEFAULT)
model_false = models.vgg16(weights=None)

preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_image = Image.open("data/test.jpg")

input_tensor = preprocess(input_image)
input_tensor = input_tensor.reshape(1, *input_tensor.shape)

if torch.cuda.is_available():
    input_tensor = input_tensor.to('cuda')
    model.to("cuda")

with torch.no_grad():
    output = model(input_tensor)

probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities)

# 获取ImageNet的标签
url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
labels = json.loads(requests.get(url).text)

# 打印前5个类别和对应的概率
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(labels[top5_catid[i]], top5_prob[i].item())
