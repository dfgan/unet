# -*- coding: utf-8 -*-
# @Time : 2021/7/27 下午9:56 
# @Author : FanDeng
# @File : Unet_block.py 
# @Software: PyCharm
import torch.nn as nn
import torch.nn.functional as F
import torch

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, mid_channel=None, kernel_size=3, activation=nn.ReLU(inplace=True)):
        super(ConvBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        if mid_channel==None:
            self.mid_channel = self.out_channel
        else:
            self.mid_channel = mid_channel
        self.kernel = kernel_size
        self.act = activation

        self.convblock = nn.Sequential(
            nn.Conv2d(self.in_channel, self.mid_channel, kernel_size=self.kernel, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_channel),
            self.act,
            nn.Conv2d(self.mid_channel, self.out_channel, kernel_size=self.kernel, stride=1, padding=1),
            nn.BatchNorm2d(self.out_channel),
            self.act,
        )

    def forward(self, x):
        return self.convblock(x)

class DownBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DownBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.down_sample = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(self.in_channel, self.out_channel)
        )

    def forward(self, x):
        return self.down_sample(x)

class UpBlock(nn.Module):
    def __init__(self, in_channel, out_channel, bilinear=True):
        super(UpBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(self.in_channel, self.in_channel//2, kernel_size=1),
            )
            self.conv = ConvBlock(self.in_channel, self.out_channel, self.in_channel//2)
        else:
            self.up = nn.ConvTranspose2d(self.in_channel, self.in_channels//2, kernel_size=2, stride=2)
            self.conv = ConvBlock(self.in_channel, self.out_channel)

    def forward(self, x_up, x_down):
        x_up = self.up(x_up)

        diffy = x_down.size()[2] - x_up.size()[2]
        diffx = x_down.size()[3] - x_up.size()[3]
        x_up = F.pad(x_up, [diffx//2, diffx-diffx//2, diffy//2, diffy-diffy//2])
        x = torch.cat([x_down, x_up], dim=1)
        return self.conv(x)