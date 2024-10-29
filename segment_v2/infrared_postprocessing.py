import PIL.Image
import cv2
import matplotlib.pyplot as plt
from PIL.Image import Image


def postprocess(original_path:str, segment_path,):

    image = cv2.imread(segment_path, cv2.IMREAD_GRAYSCALE)

    # 使用高斯模糊平滑图像
    gaussian = cv2.GaussianBlur(image, (5, 5), 0)
    ret, threshold = cv2.threshold(gaussian, 125, 255, 0)

    color_list = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

    # 查找轮廓
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 筛选出最大的三个轮廓
    if len(contours) >= 3:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
    else:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # 创建一个新的图像用于绘制轮廓
    contour_image = threshold.copy()
    contour_image = cv2.cvtColor(contour_image, cv2.COLOR_GRAY2BGR)
    # 对每个轮廓进行多边形拟合并绘制
    for i, contour in enumerate(contours):
        # 像素为单位
        area = cv2.contourArea(contour)
        if (area < 2500):
            continue
        print(f"{i}的面积:{area}")
        print(f"轮廓:{contour}" )
        epsilon = 0.002 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        cv2.drawContours(contour_image, [approx], -1, color_list[i], 10)

    # 使用matplotlib显示图像
    plt.figure(figsize=(30, 8))

    # 读取图像
    original_image = PIL.Image.open(original_path)

    plt.subplot(1, 5, 1)
    plt.title('original')
    plt.axis('off')
    plt.imshow(original_image)

    plt.subplot(1, 5, 2)
    plt.title('segment_Image')
    plt.axis('off')
    plt.imshow(image, cmap="gray")

    plt.subplot(1, 5, 3)
    plt.title('gaussian')
    plt.axis('off')
    plt.imshow(gaussian, cmap="gray")

    plt.subplot(1, 5, 4)
    plt.title('threshold')
    plt.axis('off')
    plt.imshow(threshold, cmap="gray")

    plt.subplot(1, 5, 5)
    plt.title('contour_image')
    plt.axis('off')
    plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
    plt.show()

