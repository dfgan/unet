# -*- coding: utf-8 -*-
# @Time : 2021/7/27 下午10:48 
# @Author : FanDeng
# @File : net_summry.py 
# @Software: PyCharm
import os
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torchsummary import summary
import cv2
from model.Unet import Unet

def plot_img_and_mask(mask):
    classes = mask.size[2] if len(mask.size) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    if classes > 1:
        for i in range(classes):
            ax[i+1].set_title(f'Output mask (class {i+1})')
            ax[i+1].imshow(mask[:, :, i])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()

file_path = '/home/df/project/segmentation/datasets/VOC2012/SegmentationClass'

files = os.listdir(file_path)
# h_list = []
# w_list = []
#
# for file in files:
#     img = cv2.imread(os.path.join(file_path, file))
#     h, w = img.shape[:2]
#     h_list.append(h)
#     w_list.append(w)
#
# min_h, max_h = min(h_list), max(h_list)
# min_w, max_w = min(w_list), max(w_list)
# print(min_h, max_h)
# print(min_w, max_w)

name = os.path.join(file_path, files[7])
img = cv2.imread(name)
cv2.imshow('otg', img)
img2 = cv2.resize(img, (460, 460), cv2.INTER_NEAREST_EXACT)
cv2.imshow('resize', img2)

cv2.waitKey(0)