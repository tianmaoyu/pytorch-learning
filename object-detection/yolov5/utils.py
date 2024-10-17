import cv2
import numpy as np
import  torch
from PIL import Image,ImageDraw

def draw_rectangle(image:Image.Image,labels)-> Image.Image:
    """
    画矩形
    :param image:
    :param labels:
    :return:
    """
    # image= img.copy()
    width,height= image.size[0:2]
    draw = ImageDraw.Draw(image)
    for label in labels:
        x, y, w, h = label[1:]
        x1 = (x - w / 2) * width
        y1 = (y - h / 2) * height
        x2 = (x + w / 2) * width
        y2 = (y + h / 2) * height
        rectangle = (x1, y1, x2, y2)
        # 定义矩形的边框颜色和宽度
        outline_color = 'red'
        line_width = 3
        # 在图片上画矩形
        draw.rectangle(rectangle, outline=outline_color, width=line_width)

    return image


def letterbox(img:Image.Image, labels, new_shape=(640, 640),stride=32, color=(114, 114, 114)):
    """
     调整，填充 图片
    :param img:
    :param labels:
    :param new_shape:
    :param stride:
    :param color:
    :return:
    """
    # Resize and pad image while meeting stride-multiple constraints
    img = np.array(img)
    shape = img.shape[:2]  # current shape [height, width]

    # Scale ratio (new / old)
    min_ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    new_unpad = int(round(shape[1] * min_ratio)), int(round(shape[0] * min_ratio))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
     # minimum rectangle
    dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    # 重新计算label
    height, width = shape
    new_height, new_width = img.shape[:2]
    new_labels = []
    for label in labels:
        index, x, y, w, h = label
        # Apply scaling and padding adjustments
        new_x = (x * width * min_ratio + left) / new_width
        new_y = (y * height * min_ratio + top) / new_height
        new_w = w * width * min_ratio / new_width
        new_h = h * height * min_ratio / new_height
        new_label = [index, new_x, new_y, new_w, new_h]
        new_labels.append(new_label)

    return img, new_labels

