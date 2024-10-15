import io
import os.path
import numpy
import numpy as np
from PIL import Image, ImageDraw


def read_text_label(path: str):
    if not os.path.exists(path):
        print(f"文件不存在 {path}")
        return None
    with open(path, "r") as txt_file:
        content = txt_file.read()
        return content


def get_points_from_content(content: str):
    line_list = content.splitlines()

    point_list = []
    for line in line_list:
        text_list = line.split()
        # 只有一类，   # 删除第一个标签
        label = int(text_list.pop(0))
        points = [(float(text_list[i]), float(text_list[i + 1])) for i in range(0, len(text_list), 2)]
        point_list.append(points)

    return point_list


def create_mask_image(point_list,sava_path:str, image_width=4000, image_height=3000):
    """基于给定的点列表，在指定尺寸的新图像上绘制多边形并保存"""
    if not point_list:
        print("没有点数据可以绘制")
        return

    img = Image.new('L', (image_width, image_height), 0)
    draw = ImageDraw.Draw(img)

    for list in point_list:
        new_point_list = []
        for item in list:
            x, y = item
            point = (x * image_width, y * image_height)
            new_point_list.append(point)
        draw.polygon(new_point_list, fill=255)

    img.save(sava_path)
    # point_list = np.array(point_list)


def preprocess_images(root_path: str):
    image_root = os.path.join(root_path, "images")
    label_root = os.path.join(root_path, "labels")
    mask_root = os.path.join(root_path, "masks")

    if not os.path.exists(mask_root):
        os.mkdir(mask_root)

    # image_path_list = os.listdir(image_root)
    label_path_list = os.listdir(label_root)

    label_path_list.sort()

    for file_name in label_path_list:
        if not os.path.basename(file_name).endswith(".txt"):
            continue

        label_file_path = os.path.join(label_root, file_name)
        image_base_name = file_name.replace(".txt", ".JPG")

        image_file_path = os.path.join(image_root, image_base_name)
        mask_file_path = os.path.join(mask_root, image_base_name)

        if not  os.path.exists(image_file_path):
            print(f"找不到图片{image_file_path}")
            continue

        width, height = Image.open(image_file_path).size
        try:
            context = read_text_label(label_file_path)
            points_list = get_points_from_content(context)
            create_mask_image(points_list,mask_file_path,width,height)
        except Exception as e:
            print(f"错误信息：{image_file_path}",e)


preprocess_images(r"D:\语义分割\水体标注\无人机红外-格式一")
