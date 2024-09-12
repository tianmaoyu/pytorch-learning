import os

from PIL import Image


def resize_image(image_path, output_path,scale_factor=0.25):
    """
    将图像等比缩小到指定的比例，并保存到指定路径。

    参数:
    image_path (str): 原始图像的路径。
    scale_factor (float): 缩小的比例，默认为 0.25。
    output_path (str): 缩放后图像的保存路径，默认为 'resized_image.png'。
    """
    # 打开图像
    original_image = Image.open(image_path)

    # 获取原始图像的宽度和高度
    original_width, original_height = original_image.size

    # 计算新的宽度和高度
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    # 缩放图像
    resized_image = original_image.resize((new_width, new_height), resample=Image.LANCZOS)

    # 保存结果
    resized_image.save(output_path)

    # 显示结果（可选）
    # resized_image.show()


def preprocess_images(root_path: str):
    image_root = os.path.join(root_path, "masks")
    resize_root = os.path.join(root_path, "resize_masks")

    if not os.path.exists(resize_root):
        os.mkdir(resize_root)

    # image_path_list = os.listdir(image_root)
    image_list = os.listdir(image_root)


    for file_name in image_list:
        image_path= os.path.join(image_root, file_name)
        resize_image_path = os.path.join(resize_root, file_name)
        resize_image(image_path,resize_image_path,scale_factor=0.25)




preprocess_images(r"D:\语义分割\水体标注\project-2-at-2024-09-06-17-48-376b4f93")