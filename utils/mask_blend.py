from PIL import Image
import  numpy

def composite_images(base_img_path, mask_img_path, output_path):
    # 打开底图
    base_img = Image.open(base_img_path).convert('RGBA')

    # 打开mask图像，并确保它是RGBA模式
    mask_img = Image.open(mask_img_path).convert('RGBA')

    # 获取mask图像的像素数据
    mask_data = mask_img.getdata()

    mask=numpy.array(mask_img)

    is_black= (mask[:,:,:3]==[0,0,0]).all(axis=2)
    mask[is_black, 3] = 0
    # mask[~is_black, 3] =255


    # 将mask图像中的黑色（或其他颜色）设置为透明
    # new_data = []
    # for item in mask_data:
    #     # 如果像素是纯黑，将其Alpha通道设置为0（完全透明）
    #     if item[0] == 0 and item[1] == 0 and item[2] == 0:
    #         new_data.append((0, 0, 0, 0))  # RGBA, Alpha为0
    #     else:
    #         new_data.append(item)  # 保持原样
    #
    # # 更新mask图像的像素数据
    # mask_img.putdata(new_data)

    # 使用alpha_composite方法将mask图像与底图合并
    result = Image.alpha_composite(base_img,  Image.fromarray(mask))

    # 保存结果
    result.save(output_path)


# 使用方法
base_img_path = 'src/0b8deede-071b95ed-bDJI_20230820112947_0045_W.jpeg'  # 底图路径
mask_img_path = 'src/labeled_image.png'  # mask路径
output_path = 'src/composite_result.png'  # 输出路径

composite_images(base_img_path, mask_img_path, output_path)