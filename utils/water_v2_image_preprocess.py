import os
import  shutil
import re
from typing import Any
from datetime import datetime

filename_pattern = r'(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})\.(png|jpe?g|gif)$'
date_format = '%Y-%m-%d-%H-%M-%S'


def match_date_in_path(file_path: str) -> datetime | None:
    try:
        file_name = os.path.basename(file_path).lower()
        match = re.match(filename_pattern, file_name)
        if match:
            date_str = match.group(1)
            date = datetime.strptime(date_str, date_format)
            return date
        else:
            return None
    except Exception as e:
        print(f"日期匹配错误:{file_path} \n {e}")
        return None

# 日期格式的图片，一一对应
def copy_mask_image(mask_root:str,image_root:str):

    mask_path_list = os.listdir(mask_root)
    image_path_list = os.listdir(image_root)

    mask_dic = [(mask_path, match_date_in_path(mask_path)) for mask_path in mask_path_list]
    image_dic = [(image_path, match_date_in_path(image_path)) for image_path in image_path_list]

    # 排序
    mask_dic.sort(key=lambda item: item[1])
    image_dic.sort(key=lambda item: item[1])

    mask_image_list = []
    current_index = -1

    for mask_index, (mask_path, mask_date) in enumerate(mask_dic):
        for image_index, (image_path, image_date) in enumerate(image_dic):

            if image_date <= mask_date and current_index < image_index:
                current_index = image_index
                mask_image_list.append((mask_path, image_path))

                # mask 对应  image  一一对应
                mask_ext = mask_path.split('.')[1]
                image_name = image_path.split('.')[0]
                new_image_path=image_name+'.'+mask_ext
                source_file=os.path.join(mask_root,mask_path)
                destination_file=os.path.join(mask_root,new_image_path)
                if not os.path.exists(destination_file):
                     shutil.copy2(source_file,destination_file)

    print(mask_image_list)


def find_image_test():
    mask_root = r"E:\语义分割\water_v2\water_v2\Annotations\aberlour"
    image_root = r"E:\语义分割\water_v2\water_v2\JPEGImages\aberlour"

    mask_path_list = os.listdir(mask_root)
    image_path_list = os.listdir(image_root)

    for image_name in image_path_list:

        mask_path= os.path.join(mask_root,image_name)
        mask_path_png = os.path.join(mask_root, image_name.replace("jpg","png"))

        if not os.path.exists(mask_path) and not  os.path.exists(mask_path_png):
            print(f"...{mask_path}..")


def preprocess_images(root_path):

    print(f"water_v2  path:{root_path}")
    dir_name_list = [
        "aberlour", "auldgirth", "bewdley", "cockermouth",
        "dublin", "evesham-lock", "galway-city", "holmrook",
        "keswick_greta", "worcester"
    ]

    for dir_name in dir_name_list:
        mask_root=os.path.join(root_path,"Annotations",dir_name)
        image_root = os.path.join(root_path, "JPEGImages", dir_name)
        print(mask_root)
        print(image_root)
        print("----"*8)
        copy_mask_image(mask_root,image_root)
        # shutil.rmtree()

    #需要删除的文件夹
    stream2_dir=r"Annotations\stream2\12_00_json"



# preprocess_images( r"E:\语义分割\water_v2\water_v2")




