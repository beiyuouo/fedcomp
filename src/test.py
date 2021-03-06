#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   eyes\src\main.py
# @Time    :   2022-02-21 14:57:19
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

import os
import sys

from fedhf.model.nn.unet import UNetMini

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from time import time
import argparse
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import make_grid
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

from dataset.fundus import FundusSegmentation
from net import DeepLab

from fedhf.api import opts
from fedhf.model.nn import UNet, UNetMini

opt = argparse.ArgumentParser()
opt.add_argument('--model_path', type=str, default=os.path.join('chkp', 'archive', 'fedeye-dp.pth'))
# opt.add_argument('--model_path', type=str, default=os.path.join('chkp', 'archive', 'fedasync.pth'))
# opt.add_argument('--model_path',
#  type=str,
#                  default=os.path.join('chkp', 'archive', 'fedasync-unlimit.pth'))
# opt.add_argument('--model_path', type=str, default=os.path.join('chkp', 'archive', 'fedprox.pth'))
# opt.add_argument('--model_path', type=str, default=os.path.join('chkp', 'archive', 'fedavg.pth'))

from component.metric import *

from dataset import transform as tr

composed_transforms_tr = transforms.Compose([
    tr.RandomScaleCrop(512),
    # tr.RandomRotate(),
    # tr.RandomFlip(),
    # tr.elastic_transform(),
    # tr.add_salt_pepper_noise(),
    # tr.adjust_light(),
    # tr.eraser(),
    tr.Normalize_tf(),
    tr.ToTensor()
])

composed_transforms_ts = transforms.Compose([
    tr.RandomCrop(512),
    # tr.Resize(512),
    tr.Normalize_tf(),
    tr.ToTensor()
])


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def main():
    args = opt.parse_args()
    model_path = args.model_path

    # model = DeepLab(opts().parse(['--num_classes', '2']))
    model = UNetMini(opts().parse([
        '--num_classes', '2', '--unet_n1', '4', '--unet_bilinear', '--input_c', '3', '--output_c',
        '2'
    ]))
    # model.load_state_dict(torch.load(model_path))
    # print(torch.load(model_path))
    # print(torch.load(model_path).keys())
    model.load(model_path)
    model.eval()

    train_bce_loss = []
    train_dice_loss = []
    test_bce_loss = []
    test_dice_loss = []
    train_acc = []
    test_acc = []

    for domain_id in range(1, 5):
        train_dataset = FundusSegmentation(base_dir=os.path.join('..', 'data', 'fundus'),
                                           dataset=f'Domain{domain_id}',
                                           split='train',
                                           transform=composed_transforms_tr)
        test_dataset = FundusSegmentation(base_dir=os.path.join('..', 'data', 'fundus'),
                                          dataset=f'Domain{domain_id}',
                                          split='test',
                                          transform=composed_transforms_ts)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

        _bce_loss_tr = 0.0
        _dice_loss_tr = 0.0
        _bce_loss_ts = 0.0
        _dice_loss_ts = 0.0
        _acc_tr = 0.0
        _acc_ts = 0.0

        tbar = tqdm(train_loader)
        for i, sample in enumerate(tbar):
            # print(sample.keys())
            with torch.no_grad():
                image, target = sample['image'], sample['label']
                output = model(image)
                # count nan number
                # print(torch.sum(torch.isnan(output)))
                _bce_loss_tr += nn.BCELoss()(torch.sigmoid(output), target).item()
                _dice_loss_tr += dice_loss(output, target)
                _acc_tr += accuracy(torch.sigmoid(output), target)

                tbar.set_description(
                    f'[{domain_id}] Train: {_bce_loss_tr / (i + 1):.4f} {_dice_loss_tr / (i + 1):.4f} {_acc_tr / (i + 1):.4f}'
                )

        print(
            f'[{domain_id}] Train: {_bce_loss_tr / len(train_loader):.4f} {_dice_loss_tr / len(train_loader):.4f} {_acc_tr / len(train_loader):.4f}'
        )
        train_bce_loss.append(_bce_loss_tr / len(train_loader))
        train_dice_loss.append(_dice_loss_tr / len(train_loader))
        train_acc.append(_acc_tr / len(train_loader))

        tbar = tqdm(test_loader)
        for i, sample in enumerate(tbar):
            # print(sample.keys())
            with torch.no_grad():
                image, target = sample['image'], sample['label']
                output = model(image)
                _bce_loss_ts += nn.BCELoss()(torch.sigmoid(output), target).item()
                _dice_loss_ts += dice_loss(output, target)
                _acc_ts += accuracy(torch.sigmoid(output), target)
                tbar.set_description(
                    f'[{domain_id}] Test: {_bce_loss_ts / (i + 1):.4f} {_dice_loss_ts / (i + 1):.4f} {_acc_ts / (i + 1):.4f}'
                )

        print(
            f'[{domain_id}] Test: {_bce_loss_ts / len(test_loader):.4f} {_dice_loss_ts / len(test_loader):.4f} {_acc_ts / len(test_loader):.4f}'
        )
        test_bce_loss.append(_bce_loss_ts / len(test_loader))
        test_dice_loss.append(_dice_loss_ts / len(test_loader))
        test_acc.append(_acc_ts / len(test_loader))

    train_bce_loss = np.array(train_bce_loss)
    train_dice_loss = np.array(train_dice_loss)
    test_bce_loss = np.array(test_bce_loss)
    test_dice_loss = np.array(test_dice_loss)
    train_acc = np.array(train_acc)
    test_acc = np.array(test_acc)

    print(f'Train: {train_bce_loss.mean():.8f} {train_dice_loss.mean():.8f} {train_acc.mean():.8f}')
    print(
        f'Train: {train_bce_loss.mean():.5f} {1-train_dice_loss.mean():.5f} {train_acc.mean():.5f}')
    print(f'Test: {test_bce_loss.mean():.8f} {test_dice_loss.mean():.8f} {test_acc.mean():.8f}')
    print(f'Test: {test_bce_loss.mean():.5f} {1-test_dice_loss.mean():.5f} {test_acc.mean():.5f}')


if __name__ == '__main__':
    main()