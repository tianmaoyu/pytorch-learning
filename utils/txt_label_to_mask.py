import io
import os.path
import  numpy
import numpy as np
from PIL import Image, ImageDraw


def read_text_label(path:str):
    if not os.path.exists(path):
        print(f"文件不存在 {path}")
        return None
    with open(path,"r") as txt_file:
        content= txt_file.read()
        return content



def get_points_from_content(content:str):
    line_list=content.splitlines()
    # 删除第一个标签
    label=None
    point_list=[]

    for line in line_list:
        text_list= content.split()
        if not label:
            label = int(text_list.pop(0))
        points = [(float(text_list[i]), float(text_list[i + 1])) for i in range(0, len(text_list), 2)]
        point_list.append(points)
    return point_list,label

def draw_polygon_on_image(point_list, image_width=4000, image_height=3000):
    """基于给定的点列表，在指定尺寸的新图像上绘制多边形并保存"""
    if not point_list:
        print("没有点数据可以绘制")
        return

    new_point_list=[]
    for list in  point_list:
        for item in list:
            x, y=item
            point=(x*image_width,y*image_height)
            new_point_list.append(point)

    # point_list = np.array(point_list)
    img = Image.new('L', (image_width, image_height), 0)
    draw = ImageDraw.Draw(img)
    draw.polygon(new_point_list, fill=255)
    img.save("labeled_image.png")

txt_file="0b8deede-071b95ed-bDJI_20230820112947_0045_W.txt"
content= read_text_label(txt_file)

point_list,label=get_points_from_content(content)
print(point_list)

draw_polygon_on_image(point_list)
print()