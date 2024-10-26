import torch

layer_anchors_list = torch.tensor([
    [[10, 13], [16, 30], [33, 23]],
    [[30., 61.], [62., 45.], [59., 119.]],
    [[116., 90.], [156., 198.], [373., 326.]]
])
layer_stride_list = torch.tensor([8, 16, 32])