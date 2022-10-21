import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data
from calculate_error import *
from datasets.datasets_list import MyDataset
import imageio
import imageio.core.util
from path import Path
from utils import *
from logger import AverageMeter
from model import *
from monoclip import *
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os
from PIL import Image
import torchvision.transforms.functional as F
transform = transforms.Compose([
    transforms.Resize((416 , 544)),
    transforms.ToTensor(),  # 将图像转换为张量,
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = '/home/student/DepthCLIP/DepthCLIP_code/IMAGE'
result = "/home/student/DepthCLIP/DepthCLIP_code/depthpredict_normalized"
if not os.path.exists(result):
    os.mkdir(result)
image_name = os.listdir(data_path)


model = MonoCLIP().to(device=device)
model.eval()

with torch.no_grad():
    for imagename in image_name:
        image_path=os.path.join(data_path, imagename)
        image = Image.open(image_path).convert("RGB")

        inputs_cuda= transform(image).unsqueeze(0).to(device)
        outputs_depth=model(inputs_cuda)
        inputs_cuda_flip=torch.flip(inputs_cuda,[3])
        outputs_depth_flip=model(inputs_cuda_flip)
        outputs_depth_flip=torch.flip(outputs_depth_flip,[3])
        outputs_depth=0.5*(outputs_depth+outputs_depth_flip)
        output_depth = nn.functional.interpolate(outputs_depth, size=[416, 544], mode='bilinear', align_corners=True)
        # output_depth = output_depth / 255
        output_depth_image = F.to_pil_image(output_depth.cpu().squeeze())

        output_path=os.path.join(result,imagename)
        output_depth_image.save(output_path)




