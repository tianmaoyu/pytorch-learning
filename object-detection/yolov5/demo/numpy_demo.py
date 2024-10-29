import numpy as np

# 创建一个示例 NumPy 数组
data = np.array([1.123456789, 2.987654321, 3.141592653589793, 4.567890123456])

print(data.round(5))
# 保留 5 位小数
rounded_data = np.round(data, 5)

print(rounded_data)
