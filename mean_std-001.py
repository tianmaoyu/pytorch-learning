from PIL import Image
import numpy as np

def calculate_mean_std_v2(image_path):
    image = Image.open(image_path)

    # 将图像转换为 NumPy 数组
    np_image = np.array(image)

    R = np_image[:, :, 0]
    G = np_image[:, :, 1]
    B = np_image[:, :, 2]

    R_mean, R_std = np.mean(R), np.std(R)
    G_mean, G_std = np.mean(G), np.std(G)
    B_mean, B_std = np.mean(B), np.std(B)

    mean=np.array([R_mean,G_mean,B_mean])
    std=np.array([R_std,G_std,B_std])
    print(mean,std)

    import torch

    mean=torch.from_numpy(mean)
    probabilities = torch.nn.functional.softmax(mean, dim=0)
    print(probabilities)

    std=torch.from_numpy(std)
    probabilities = torch.nn.functional.softmax(std, dim=0)
    print(probabilities)



def calculate_mean_std(image_path):
    # 加载图像
    image = Image.open(image_path)

    # 将图像转换为 NumPy 数组
    np_image = np.array(image)

    # 检查图像是否有三个通道（RGB）
    if np_image.ndim != 3 or np_image.shape[2] != 3:
        raise ValueError("图像需要有三个通道（RGB）")

        # 分离 RGB 通道
    R = np_image[:, :, 0]
    G = np_image[:, :, 1]
    B = np_image[:, :, 2]

    # 计算每个通道的均值和标准差
    R_mean, R_std = np.mean(R), np.std(R)
    G_mean, G_std = np.mean(G), np.std(G)
    B_mean, B_std = np.mean(B), np.std(B)
    B_mean, B_std = np.mean(B), np.std(B)


    return {
        "R": {"mean": R_mean, "std": R_std},
        "G": {"mean": G_mean, "std": G_std},
        "B": {"mean": B_mean, "std": B_std}
    }


# 示例用法
image_path = "data/test.jpg"
mean_std = calculate_mean_std(image_path)
calculate_mean_std_v2(image_path)
print(mean_std)


