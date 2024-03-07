# -*- coding:utf-8 -*-
"""
# @Author  :Shan Huang
# @Time    :2024/03/05 15:04 
# @File    :kitti_select.py
"""
import os

output_path = 'kitti_train_selection.txt'
image_directory_path = r'Z:\迅雷下载\KITTI_depth_completion\depth_selection\val_selection_cropped\image'
depth_directory_path = r'Z:\迅雷下载\KITTI_depth_completion\depth_selection\val_selection_cropped\groundtruth_depth'

def get_all_files_in_directory(directory):
    files = []
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        if os.path.isfile(path):
            files.append(path)
    return files

def split_file_names(files):
    split_files = []
    for file_path in files:
        # 使用'_'分割文件名
        split_file_name = os.path.basename(file_path).split('_')
        split_files.append(split_file_name)
    return split_files

image_name = get_all_files_in_directory(image_directory_path)
depth_name = get_all_files_in_directory(depth_directory_path)
image_name_split = split_file_names(image_name)
depth_name_split = split_file_names(depth_name)
img_test = []
dep_test = []
for i in image_name_split:
    img_test.append(i[7])
for i in depth_name_split:
    dep_test.append(i[8])

match_file_name = []
for file_name in image_name:
    ima_path = os.path.basename(file_name)
    ima_path_split = ima_path.split('_')
    ima_path_split[6] = 'groundtruth'
    ima_path_split.insert(7, 'depth')
    dep_name = '_'.join(ima_path_split)
    p=os.path.join(depth_directory_path, dep_name)
    if os.path.isfile(os.path.join(depth_directory_path, dep_name)):
        match_file_name.append((os.path.basename(file_name), os.path.basename(dep_name)))

print(len(match_file_name))
with open('kitti_selection_train.txt', 'w') as file:
    for tpl in match_file_name:
        line = ' '.join(tpl)
        file.write(line+'\n')











