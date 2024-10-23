import torch
import torchvision
from  torchvision.ops import  ciou_loss,distance_box_iou,box_iou

# 定义候选框，形状为 [N, 4]，每个框 [x1, y1, x2, y2]
boxes = torch.tensor([[10, 10, 20, 20],  # 第一个框
                      [12, 12, 22, 22],  # 第二个框，和第一个框有重叠
                      [100, 100, 160, 160],  # 第三个框
                      [110, 110, 160, 160]],dtype=torch.float)  # 第四个框，和第三个框有重叠

# 定义每个候选框的置信度得分
scores = torch.tensor([0.9, 0.75, 0.8, 0.6],dtype=torch.float)


box_1=torch.tensor([[10, 10, 20, 20]],dtype=torch.float)
box_2=torch.tensor([[12, 12, 22, 22]],dtype=torch.float)
iou= box_iou(boxes,boxes)
# 调用 NMS
indices = torchvision.ops.nms(boxes, scores, 0.5)

boxes_1 = torch.tensor([[10, 10, 20, 20],
                      [12, 12, 22, 22],
                      [100, 100, 150, 150],
                      [110, 110, 160, 160]],dtype=torch.float)
boxes_2 = torch.tensor([[10, 10, 20, 20],
                      [12, 12, 22, 22],
                      [100, 100, 150, 150],
                      [110, 110, 160, 160]],dtype=torch.float)

ciou_loss=ciou_loss.complete_box_iou_loss(boxes_1,boxes_2)

# 输出：tensor([0, 2])，保留了第一个和第三个框
print(indices)