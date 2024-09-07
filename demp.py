import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')  # 或者 'Qt5Agg'
# 示例图片列表，替换为你自己的图片路径
image_paths = ["./imgs/1.PNG", "./imgs/2.PNG", "./imgs/3.PNG",
               "./imgs/4.PNG", "./imgs/5.PNG", "./imgs/5.PNG"]

# 创建一个绘图窗口
fig, ax = plt.subplots()
# 遍历图片并显示
for i in tqdm(range(len(image_paths)), desc="加载图片"):
    # 清除之前的图片
    ax.clear()
    # 打开图片并显示
    img = Image.open(image_paths[i])
    ax.imshow(img)
    # ax.axis('off')  # 关闭坐标轴
    # 显示当前的图片
    plt.draw()
    plt.pause(1)  # 设置暂停时间以控制每张图片显示的时长
    img.close()


plt.show()