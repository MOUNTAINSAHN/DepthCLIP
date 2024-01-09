# -*- coding:utf-8 -*-
"""
# @Author  :Shan Huang
# @Time    :2023/12/05 22:08 
# @File    :readnpy.py
"""
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
data = np.load(r"D:\DepthCLIP\DepthCLIP_code\cityscapses_depth\train\depth\1.npy")
# data = np.array(Image.open(r"D:\DepthCLIP\DepthCLIP_code\datasets\NYU_Depth_V2\official_splits\test\bathroom\dense\sync_depth_dense_00045.png"))
print(np.max(data))
print(np.min(data))

# tensor_data = torch.from_numpy(data)
# tensor_data = torch.permute(2, 0, 1)
# data = np.squeeze(data, axis=-1)

data_normalized = (data*255).astype(np.uint8)
image = Image.fromarray(data_normalized)
image.show()
image.save("label1.jpg")
