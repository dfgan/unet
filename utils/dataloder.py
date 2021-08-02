# -*- coding: utf-8 -*-
# @Time : 2021/7/27 下午10:53 
# @Author : FanDeng
# @File : dataloder.py 
# @Software: PyCharm
import os
from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import random
import logging
import matplotlib.pyplot as plt
import cv2

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, resize=(460, 460), infer_size=(320, 320), mask_suffix=''):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.resize = resize
        self.crop_size = infer_size
        assert 0 < scale <= 1, "Scale must between 0 and 1"

        self.color2label, self.label2color, self.classes = color_label()

        image_names = [splitext(file)[0] for file in listdir(imgs_dir) if not file.startswith('.')]
        mask_names = [splitext(file)[0] for file in listdir(masks_dir) if not file.startswith('.')]

        self.ids = list(set(image_names) & set(mask_names))
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, "Scale is too small!"

        pil_img = pil_img.resize((newW, newH))
        img_nd = np.array(pil_img)
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255
        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = os.path.join(self.masks_dir, idx + '.png')
        img_file = os.path.join(self.imgs_dir, idx + '.jpg')

        print(mask_file)

        mask = cv2.cvtColor(cv2.imread(mask_file), cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)

        img, label = self.train_transform(img, mask, crop_size=self.crop_size)
        return {
            'image': img,
            'mask': label
        }

    # def __getitem__(self, i):
    #     idx = self.ids[i]
    #     mask_file = os.path.join(self.masks_dir, idx + '.png')
    #     img_file = os.path.join(self.imgs_dir, idx + '.jpg')
    #
    #     # assert len(mask_file) == 1, \
    #     #     f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
    #     # assert len(img_file) == 1, \
    #     #     f'Either no image or multiple images found for the ID {idx}: {img_file}'
    #
    #     mask = Image.open(mask_file).convert('RGB')
    #     img = Image.open(img_file).convert('RGB')
    #
    #     mask = mask.resize(self.img_size, Image.BILINEAR)
    #     img = img.resize(self.img_size, Image.BILINEAR)
    #
    #     img = self.preprocess(img, self.scale)
    #     mask = self.preprocess(mask, self.scale)
    #
    #     return {
    #         'image': torch.from_numpy(img).type(torch.FloatTensor),
    #         'mask': torch.from_numpy(mask).type(torch.FloatTensor)
    #     }

    def train_transform(self, img, label, crop_size=(256, 256)):
        label1 = self.label_map(label)

        label = cv2.resize(label1, self.resize, cv2.INTER_NEAREST)
        # fig = plt.figure(figsize=(60, 30))
        # ax1 = fig.add_subplot(1, 2, 1)
        # ax1.imshow(label1)
        # ax1.set_title('RGB')
        #
        # ax2 = fig.add_subplot(1, 2, 2)
        # ax2.imshow(label)
        # ax2.set_title('mask')
        # plt.show()
        img = cv2.resize(img, self.resize)
        img, label = RandomCrop(img, label, crop_size)

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
        img = transform(img)
        return img, torch.from_numpy(label).long()

    def label_map(self, label):
        label = np.array(label, dtype=np.int64)
        map = np.zeros(label.shape[:2])
        for i in range(label.shape[0]):
            for j in range(label.shape[1]):
                try:
                    map[i][j] = self.color2label[str(list(label[i][j]))]
                except:
                    map[i][j] = 0
        return map
    
class VOCDataset(BasicDataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super(VOCDataset, self).__init__(imgs_dir, masks_dir, scale, mask_suffix='')

def color_label():
    classes = ['background','aeroplane','bicycle','bird','boat',
               'bottle','bus','car','cat','chair','cow','diningtable',
               'dog','horse','motorbike','person','potted plant',
               'sheep','sofa','train','tv/monitor']
    # 给每一类都来一种颜色
    colormap = [[0,0,0],[128,0,0],[0,128,0], [128,128,0], [0,0,128],
                [128,0,128],[0,128,128],[128,128,128],[64,0,0],[192,0,0],
                [64,128,0],[192,128,0],[64,0,128],[192,0,128],
                [64,128,128],[192,128,128],[0,64,0],[128,64,0],
                [0,192,0],[128,192,0],[0,64,128]]

    color2label = {}
    label2color = {}

    for i, value in enumerate(colormap):
        color2label[str(value)] = i
        if i not in label2color:
            label2color[i] = value

    return color2label, label2color, classes


def RandomCrop(img, label, size):
    h, w = img.shape[:2]
    th, tw = size
    assert h >= th and w >= th, 'this is wrong crop size'

    if w == tw and h == th:
        i, j = 0, 0
    else:
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        h, w = th, tw
    return img[i:i+h, j:j+w, :], label[i:i+h, j:j+w]