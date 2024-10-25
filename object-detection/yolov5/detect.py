import PIL.Image
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch import Tensor
from torchvision.transforms import ToTensor
import utils


def xywh2xyxy(data: Tensor):
    temp = data.clone()
    x1 = temp[..., 0] - temp[..., 2] / 2  # x1 = center_x - width / 2
    y1 = temp[..., 1] - temp[..., 3] / 2  # y1 = center_y - height / 2
    x2 = temp[..., 0] + temp[..., 2] / 2  # x2 = center_x + width / 2
    y2 = temp[..., 1] + temp[..., 3] / 2  # y2 = center_y + height / 2

    data[..., 0] = x1
    data[..., 1] = y1
    data[..., 2] = x2
    data[..., 3] = y2
    return data


def draw_image(data: Tensor, image_path):
    image = PIL.Image.open(image_path)
    boxes = data[:5, :4]

    plt.figure(dpi=300)

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    img_draw = utils.draw_rectangle_xyxy(image, boxes.cpu().numpy())
    plt.imshow(img_draw)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

layer_anchors_list = torch.tensor([
    [[10, 13], [16, 30], [33, 23]],
    [[30., 61.], [62., 45.], [59., 119.]],
    [[116., 90.], [156., 198.], [373., 326.]]
])
layer_stride_list = torch.tensor([8, 16, 32])

model_path = "./out/yolov5-229.pth"
image_path = "./out/img_4.png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image = Image.open(image_path).convert("RGB")
model = torch.load(model_path, map_location=device, weights_only=False)
# model= torch.load(model_path,weights_only=False)
model.eval()
with torch.no_grad():
    image = ToTensor()(image)
    image = image.unsqueeze(dim=0)
    # 填充最小能被32整除
    image = utils.image_pad(image, scale=32)
    image = image.to(device)
    layer_list = model(image)

    output_list = []
    #
    for i, layer in enumerate(layer_list):
        layer_stride = layer_stride_list[i]
        layer_anchor = layer_anchors_list[i]

        # 变形 [bs,3*(5+class_num),h,w] ->  [bs,3, (5+class_num),h,w] ->  [bs,3,h,w,(5+class_num)]
        bs, channel, height, width = layer.shape

        data = layer.view(bs, 3, channel // 3, height, width).permute(0, 1, 3, 4, 2).contiguous()
        # predict=torch.sigmoid(predict)

        grid_y, grid_x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        #  [ny, nx, 2]  -> [1,1,ny,nx,2]
        grid_xy = torch.stack([grid_x, grid_y], 2).view(1, 1, height, width, 2).float()
        # [3,2]->[1,3,1,1,2]
        anchor_wh = layer_anchor.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        data = torch.sigmoid(data)
        xy = data[..., :2]
        wh = data[..., 2:4]
        cls = data[..., 5:]

        output = torch.zeros([bs, 3, height, width, 6], device=device)
        score, label_index = torch.max(cls, dim=4, keepdim=True)
        # 还原到原图片大小
        output[..., 0:2] = (xy * 2.0 - 0.5 + grid_xy) * layer_stride
        output[..., 2:4] = (wh * 2.0) ** 2 * anchor_wh
        output[..., 4:5] = score
        output[..., 5:6] = label_index

        # [bs,3,h,w,6] -> [bs,-1,6] 因为 检测时 bs=1  因为 batched_nms
        output = output.view(-1, 6)
        output_list.append(output)

    output = torch.cat(output_list, dim=0)
    output = output[(output[..., 4] > 0.25) & (output[..., 2] > 2) & (output[..., 3] > 2)]
    # ba x
    output = xywh2xyxy(output)
    indices = torchvision.ops.batched_nms(boxes=output[..., :4],
                                          scores=output[..., 4],
                                          idxs=output[..., 5],
                                          iou_threshold=0.45)
    filter_data = output[indices]
    # filter_data = filter_data[filter_data[:,5]==22.0]
    print(filter_data)
    draw_image(filter_data,image_path)






