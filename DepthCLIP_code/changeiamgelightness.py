from PIL import Image
import numpy as np

# 打开PNG图像
imagepath=('/home/student/DepthCLIP/DepthCLIP_code/datasets/NYU_Depth_V2/official_splits/test/bathroom/dense/sync_depth_dense_00045.png')
image = Image.open(imagepath)

# 将图像转换为NumPy数组
image_array = np.array(image)
# 现在，image_array 包含了PNG图像的像素数据
normalized_image_array = image_array / image_array.max()
normalized_image = Image.fromarray((normalized_image_array * 255).astype('uint8'))
normalized_image.save('normalized_image3.png')
normalized_image.show()