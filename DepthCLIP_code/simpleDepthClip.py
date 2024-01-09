# -*- coding:utf-8 -*-
"""
# @Author  :Shan Huang
# @Time    :2023/12/20 17:06 
# @File    :simpleDepthClip.py
"""
import clip
import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameter Control:
depth_templates = ['This {} is {}']
obj_classes = ['object']
depth_classes = ['giant', 'extremely close', 'close', 'not in distance', 'a little remote', 'far', 'unseen']
bin_list = [1.00, 1.50, 2.00, 2.25, 2.50, 2.75, 3.00]
temperature = 0.1
clip_vis = 'RN50'


def zeroshot_classifier(depth_classes, obj_classes, templates, model):
    with torch.no_grad():
        zeroshot_weights = []
        for depth in depth_classes:
            for obj in obj_classes:
                texts = [template.format(obj, depth) for template in templates]  # format with class
                texts = clip.tokenize(texts).cuda()  # tokenize
                class_embeddings = model.encode_text(texts)  # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


class FCLayer(nn.Module):
    def __init__(self, c_in=1024, reduction=4):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


# CLIP for Monocular Depth Estimation
class MonoCLIP(nn.Module):
    def __init__(self):
        super(MonoCLIP, self).__init__()
        self.bins = len(depth_classes)

        self.clip, _ = clip.load(clip_vis)  # load pretrained clip encoder
        self.text_f = zeroshot_classifier(depth_classes, obj_classes, depth_templates, self.clip)  # init text feature
        self.adapter = FCLayer(1024).to(self.clip.dtype)

    def forward(self, x):
        img_f = self.clip.encode_image(x).permute(1, 0, 2)  # B, HW, C; shape:[1,221,2048]
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)  # normalize img_f

        # @: dot product of two vectors
        img_f = torch.nn.functional.interpolate(img_f, scale_factor=0.5)  # to match size; shape: [1,221,1024]

        depth_logits = 100. * img_f @ self.text_f  # B, HW, K # img_f and text_f have both been normalized, so just use a inner product
        depth_logits = depth_logits.permute(0, 2, 1).reshape(-1, self.bins, 13, 17)  # B, K, H, W
        depth_logits /= temperature

        depth = F.softmax(depth_logits, dim=1)  # shape:[1,7,13,17]
        # bin_tensor = torch.tensor(bin_list).to(depth.device)  # shape: [7]
        # depth = depth * bin_tensor.reshape(1, self.bins).unsqueeze(-1).unsqueeze(-1)  # shape:[1,7,13,17] * [1,7,1,1]
        # depth = depth.sum(1, keepdim=True)
        return depth  # shape: [1,1,13,17]


class UnfixedValues(nn.Module):
    def __init__(self):
        super(UnfixedValues, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=7, out_channels=1, kernel_size=1, stride=1, padding=0, device="cuda:0")

    def forward(self, x):
        x = self.conv2d(x)
        x = F.relu(x)
        return x


class UnfixedDepthValues(nn.Module):
    def __init__(self):
        super(UnfixedDepthValues, self).__init__()
        self.binlist = nn.Parameter(
            torch.tensor([1.00, 1.50, 2.00, 2.25, 2.50, 2.75, 3.00]).view([1, 7, 1, 1]).to(device="cuda:0"))
        self.conv2d = nn.Conv2d(in_channels=7, out_channels=1, kernel_size=1, stride=1, padding=0, device="cuda:0")

    def forward(self, x):
        x = x * self.binlist
        x = self.conv2d(x)
        x = F.relu(x)
        return x



class IntergratedDepthCLIP(nn.Module):
    def __init__(self):
        super(IntergratedDepthCLIP, self).__init__()
        self.monoclip = MonoCLIP()
        for param in self.monoclip.parameters():
            param.requires_grad = False
        # self.unfixedvalue = UnfixedValues().cuda().half()
        self.unfixedvalue = UnfixedDepthValues().cuda().half()
        for param in self.monoclip.parameters():
            param.requires_grad = True


    def forward(self, x):
        x_flip = torch.flip(x, [3])
        output_depth = self.monoclip(x)
        output_depth = self.unfixedvalue(output_depth)
        output_depth_flip = self.monoclip(x_flip)
        output_depth_flip = self.unfixedvalue(output_depth_flip)
        output_depth_flip = torch.flip(output_depth_flip, [3])
        output_depth = 0.5 * (output_depth + output_depth_flip)
        output_depth = nn.functional.interpolate(output_depth, size=[416, 544], mode='bilinear', align_corners=True)
        return output_depth


class DepthLoss(nn.Module):
    def __init__(self):
        super(DepthLoss, self).__init__()

    def forward(self, volume, gt):
        cos = torch.nn.CosineSimilarity(dim=3, eps=1e-08)
        batch = volume.shape[0]
        dim = volume.shape[2]
        losses3 = torch.sum((gt - volume).abs()) / (batch * dim * dim * 1)
        part1 = torch.sum(torch.abs(1 - cos(volume, gt)))
        part2 = torch.sum(torch.abs(1 - cos(volume.transpose(2, 3), gt.transpose(2, 3))))
        losses3 = losses3 + (part1 + part2) / (batch * dim * 1)
        return losses3
