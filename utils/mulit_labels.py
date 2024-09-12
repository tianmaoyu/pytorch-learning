# 标签中每个RGB颜色的值
import numpy as np
import torch
from numpy.array_api import uint8
from torchvision.transforms import ToPILImage
from torchvision.transforms.v2 import ToImage

VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]
# 标签其标注的类别
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']
colormap2label = torch.zeros(256**3, dtype=torch.uint8) # torch.Size([16777216])
for i, colormap in enumerate(VOC_COLORMAP):
    # 每个通道的进制是256，这样可以保证每个 rgb 对应一个下标 i
    colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i

# 构造标签矩阵
def voc_label_indices(colormap, colormap2label):
    colormap = np.array(colormap.convert("RGB")).astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256 + colormap[:, :, 2])
    return colormap2label[idx] # colormap 映射 到colormaplabel中计算的下标


image_tensor = torch.randint(0,255,(3, 512, 512),dtype=torch.uint8)  # 生成一个 3xHxW 的随机张量，代表 RGB 三个通道
# 将张量转换为 PIL 图像
image = ToPILImage()(image_tensor)
y = voc_label_indices(image, colormap2label)
print(y[0:110, 0:140]) #打印结果是一个int型tensor，tensor中的每个元素i表示该像素的类别是VOC_CLASSES[i]
