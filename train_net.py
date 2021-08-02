# -*- coding: utf-8 -*-
# @Time : 2021/7/27 下午11:17 
# @Author : FanDeng
# @File : train_net.py 
# @Software: PyCharm

import argparse
import logging
import os
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split

from model.Unet import Unet
from utils.dataloder import VOCDataset
from tools.eval import eval_net, label_accuracy_score

def train(net,
          device,
          epochs=20,
          batch_size=5,
          lr=0.001,
          img_path='',
          mask_path='',
          val_percent=0.1,
          save_cp=True,
          img_scale=0.5):

    dataset = VOCDataset(img_path, mask_path, img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=10)
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.BCEWithLogitsLoss().cuda()

    check_loss = float('inf')

    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        label_true=torch.LongTensor()
        label_pred=torch.LongTensor()
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}',unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                gt_mask = batch['mask']
                assert imgs.shape[1] == net.in_channel, \
                    f'Network has been defined with {net.in_channel} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = gt_mask.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                writer.add_scalar('loss/train', loss.item(), global_step)
                pbar.set_postfix(**{'loss (batch)':loss.item()})

                pred = masks_pred.argmax(dim=1).squeeze().data.cpu()
                real = gt_mask.data.cpu()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                label_true=torch.cat((label_true,real),dim=0)
                label_pred=torch.cat((label_pred,pred),dim=0)

                global_step += 1

                show_img = imgs[-1].permute(1,2,0).cpu().numpy()
                show_mask = gt_mask[-1].cpu().numpy()
                # cv2.imshow('RGB', show_img)
                # cv2.imshow('mask', show_mask)
                # cv2.waitKey(0)


            acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(label_true.numpy(), label_pred.numpy(), net.n_classes)
            print('Training!!!! the pred ave acc is {}, the class acc is {}, the mean iou is {}'.format(acc, acc_cls, mean_iu))
            if val_percent>0:
            # if global_step % (n_train // (50 * batch_size)) == 0:
                for tag, value in net.named_parameters():
                    tag = tag.replace('.', '/')
                    writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                    writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                val_score = eval_net(net, val_loader, device)
                scheduler.step(val_score)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                if net.n_classes > 1:
                    logging.info('Validation cross entropy:{}'.format(val_score))
                    writer.add_scalar('Loss/test', val_score, global_step)
                else:
                    logging.info('Validation Dice Coeff: {}'.format(val_score))
                    writer.add_scalar('Dice/test', val_score, global_step)

                writer.add_images('images', imgs, global_step)
                if net.n_classes == 1:
                    writer.add_images('masks/true', true_masks, global_step)
                    writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}_{val_score}.path')
            logging.info(f'Checkpoint {epoch + 1} saved!')
            if val_score < check_loss:
                check_loss = val_score
                torch.save(net.state_dict(),
                           dir_checkpoint + f'best_model.path')
    writer.close()

if __name__ == "__main__":
    data_images = '/home/df/project/segmentation/datasets/VOC2012/JPEGImages'
    data_masks = '/home/df/project/segmentation/datasets/VOC2012/SegmentationClass'
    dir_checkpoint = '../checkpoints'
    check_path = os.path.join(dir_checkpoint, 'best_model.path')


    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'using device {device}')

    net = Unet(in_channel=3, out_classes=21, bilinear=True)
    logging.info(f'Network:\n'
                 f'\t{net.in_channel} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    try :
        net.load_state_dict(
            torch.load(check_path, map_location=device)
        )
        logging.info(f"loading model weight from {check_path}")
    except:
        pass

    net.to(device)
    train(net,
          epochs=100,
          batch_size=4,
          lr=0.001,
          img_path=data_images,
          mask_path=data_masks,
          device=device,
          img_scale=0.5,
          val_percent=0.1,
          save_cp=True)