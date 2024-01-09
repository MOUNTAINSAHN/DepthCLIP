# -*- coding:utf-8 -*-
"""
# @Author  :Shan Huang
# @Time    :2024/01/09 12:28 
# @File    :draw.py
"""
import os
import matplotlib.pyplot as plt
import pandas as pd
log_file_path = r'D:\DepthCLIP\DepthCLIP_code\training_log.txt'
log_data=pd.read_csv(log_file_path, sep=' - ', engine='python', header=None, names=['Time', 'Level', 'Message'])
log_data['Time'] = pd.to_datetime(log_data['Time'], format='%Y-%m-%d %H:%M:%S,%f')
learning_rate_data = log_data[log_data['Message'].str.contains('Learning Rate')]
loss_data = log_data[log_data['Message'].str.contains('Loss')]

# plt.figure(figsize=(5, 5))
# plt.plot(range(1, 21), learning_rate_data['Message'].str.extract(r'(?<=Learning Rate: )(\d+\.\d+)').astype(float), label='Learning Rate')
# # plt.plot(learning_rate_data['index'], learning_rate_data['Message'].str.extract(r'(?<=Learning Rate: )(\d+\.\d+)').astype(float), label='Learning Rate')
# plt.title('Learning Rate Curve')
# plt.xlabel('Epoch')
# plt.ylabel('Learning Rate')
# plt.xticks(range(1, 21))
# plt.legend()
# plt.show()

# # 绘制损失曲线
# plt.figure(figsize=(5, 5))
# plt.plot(range(1, 21), loss_data['Message'].str.extract(r'(?<=Loss: )(\d+\.\d+)').astype(float), label='Loss')
# plt.title('Loss Curve')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.xticks(range(1, 21))
# plt.legend()
# plt.show()

# plt.figure(figsize=(5, 5))
# plt.plot(range(1, 21), loss_data['Message'].str.extract(r'(?<=abs_diff : )(\d+\.\d+)').astype(float), label='abs_diff')
# plt.title('abs_diff')
# plt.xlabel('Epoch')
# plt.ylabel('abs_diff')
# plt.xticks(range(1, 21))
# plt.legend()
# plt.show()
#
# plt.figure(figsize=(5, 5))
# plt.plot(range(1, 21), loss_data['Message'].str.extract(r'(?<=a1 : )(\d+\.\d+)').astype(float), label='a1')
# plt.title('a1')
# plt.xlabel('Epoch')
# plt.ylabel('a1')
# plt.xticks(range(1, 21))
# plt.legend()
# plt.show()
#
# plt.figure(figsize=(5, 5))
# plt.plot(range(1, 21), loss_data['Message'].str.extract(r'(?<=a2 : )(\d+\.\d+)').astype(float), label='a2')
# plt.title('a2')
# plt.xlabel('Epoch')
# plt.ylabel('a2')
# plt.xticks(range(1, 21))
# plt.legend()
# plt.show()

# plt.figure(figsize=(5, 5))
# plt.plot(range(1, 21), loss_data['Message'].str.extract(r'(?<=a2 : )(\d+\.\d+)').astype(float), label='a3')
# plt.title('a3')
# plt.xlabel('Epoch')
# plt.ylabel('a3')
# plt.xticks(range(1, 21))
# plt.legend()
# plt.show()

# plt.figure(figsize=(5, 5))
# plt.plot(range(1, 21), loss_data['Message'].str.extract(r'(?<=abs_rel : )(\d+\.\d+)').astype(float), label='abs_rel')
# plt.title('abs_rel')
# plt.xlabel('Epoch')
# plt.ylabel('abs_rel')
# plt.xticks(range(1, 21))
# plt.legend()
# plt.show()

# plt.figure(figsize=(5, 5))
# plt.plot(range(1, 21), loss_data['Message'].str.extract(r'(?<=log10 : )(\d+\.\d+)').astype(float), label='log10')
# plt.title('log10')
# plt.xlabel('Epoch')
# plt.ylabel('log10')
# plt.xticks(range(1, 21))
# plt.legend()
# plt.show()

plt.figure(figsize=(5, 5))
plt.plot(range(1, 21), loss_data['Message'].str.extract(r'(?<=rmse : )(\d+\.\d+)').astype(float), label='rmse')
plt.title('rmse')
plt.xlabel('Epoch')
plt.ylabel('rmse')
plt.xticks(range(1, 21))
plt.legend()
plt.show()


