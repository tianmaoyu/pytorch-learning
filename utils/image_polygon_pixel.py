from PIL import Image
import  numpy

def polygon_images(base_img_path, mask_img_path, output_path):
    # 打开底图
    base_img = Image.open(base_img_path).convert('RGBA')
    # 打开mask图像，并确保它是RGBA模式
    mask_img = Image.open(mask_img_path).convert('L')
    # 转换为numpy数组以便操作
    base_array = numpy.array(base_img)
    mask_array = numpy.array(mask_img)
    # 使用掩码来设置透明度
    # 将mask值为255的位置对应的alpha通道设为0（完全透明）
    # base_array[:, :, 3][mask_array == 255] = 0
    base_array[:, :, 3][mask_array != 255] = 0

    # 从修改后的numpy数组创建新图像
    result_img = Image.fromarray(base_array)

    # 保存结果
    result_img.save(output_path)

# 使用方法
base_img_path = 'src/0b8deede-071b95ed-bDJI_20230820112947_0045_W.jpeg'  # 底图路径
mask_img_path = 'src/labeled_image.png'  # mask路径
output_path = 'src/polygon_result.png'  # 输出路径

polygon_images(base_img_path, mask_img_path, output_path)