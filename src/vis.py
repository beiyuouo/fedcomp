#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   eyes\src\main.py
# @Time    :   2022-02-21 14:57:19
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from time import time
import argparse
from tqdm import tqdm

import numpy as np

import torch
from torchvision import transforms
from torchvision.utils import make_grid
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

from dataset.fundus import FundusSegmentation
from net import DeepLab

from fedhf.api import opts
from fedhf.model.nn import UNet

opt = argparse.ArgumentParser()
opt.add_argument('--model_path', type=str, default=os.path.join('chkp', 'best.pth'))

from component.metric import DiceLoss

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
    model = UNet(opts().parse(
        ['--num_classes', '2', '--unet_n1', '16', '--input_c', '3', '--output_c', '2']))
    model.load(model_path)
    model.eval()

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

        criterion = DiceLoss()

        # train_loss = 0.0
        # test_loss = 0.0

        # tbar = tqdm(train_loader)
        # for i, sample in enumerate(tbar):
        #     # print(sample.keys())
        #     with torch.no_grad():
        #         image, target = sample['image'], sample['label']
        #         output = model(image)
        #         loss = criterion(torch.sigmoid(output), target)
        #         train_loss += loss.item()
        #         tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))

        # print(f'Domain{i} Train loss: {train_loss / len(train_loader)}')

        # tbar = tqdm(test_loader)
        # for i, sample in enumerate(tbar):
        #     # print(sample.keys())
        #     with torch.no_grad():
        #         image, target = sample['image'], sample['label']
        #         output = model(image)
        #         loss = criterion(torch.sigmoid(output), target)
        #         test_loss += loss.item()
        #         tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))

        # print(f'Domain{i} Test loss: {test_loss / len(test_loader)}')

        tbar = tqdm(test_loader)
        for i, sample in enumerate(tbar):
            with torch.no_grad():
                image, target = sample['image'], sample['label']
                output = model(image)
                output = torch.sigmoid(output)
                # print(output.shape)

                gt_img = make_grid([target[0][0], target[0][1]])
                pred_img = make_grid([output[0][0], output[0][1]])

                # print(gt_img.shape)
                # print(pred_img.shape)

                # print(gt_img.unique())
                # print(pred_img.unique())

                show([image[0], gt_img[0], gt_img[1], pred_img[0], pred_img[1]])

                plt.show()
                plt.savefig(os.path.join('log', 'vis', f'Domain{domain_id}_test{i}.png'))
                plt.close()


if __name__ == '__main__':
    main()