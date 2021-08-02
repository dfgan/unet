# -*- coding: utf-8 -*-
# @Time : 2021/7/27 下午10:37 
# @Author : FanDeng
# @File : Unet.py 
# @Software: PyCharm
from .Unet_block import *

import torch
import torch.nn as nn
import torch.nn.functional as F

class Unet(nn.Module):
    def __init__(self, in_channel=3, out_classes=21, bilinear=True):
        super(Unet, self).__init__()
        self.in_channel = in_channel
        self.n_classes = out_classes
        self.bilinear = bilinear

        self.in_put = ConvBlock(self.in_channel, 64)
        self.down1 = DownBlock(64, 128)
        self.down2 = DownBlock(128, 256)
        self.down3 = DownBlock(256, 512)
        self.down4 = DownBlock(512, 1024)

        self.up1 = UpBlock(1024, 512)
        self.up2 = UpBlock(512, 256)
        self.up3 = UpBlock(256, 128)
        self.up4 = UpBlock(128, 64)

        self.out_put = nn.Conv2d(64, self.n_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x1 = self.in_put(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x = self.out_put(x)
        return F.log_softmax(x, dim=1)