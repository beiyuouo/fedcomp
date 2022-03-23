#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   experiments\fedeye\src\component\evalutor.py
# @Time    :   2022-03-07 16:51:18
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

import os
import sys
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import fedhf
from fedhf.component.evaluator import BaseEvaluator

from .metric import *


class Evaluator(BaseEvaluator):

    def __init__(self, args):
        super(Evaluator, self).__init__(args)

    def evaluate(self, dataloader, model, client_id=None, gpus=[], device='cpu'):
        if not client_id:
            client_id = -1
        self.model = model.to(device)
        self.criterion = nn.BCELoss()
        # self.criterion = dice_loss
        self.device = device

        self.logger.info(f'Start evaluation on {client_id}')

        model.eval()
        tbar = tqdm(dataloader)
        num_img_tr = len(dataloader)
        test_bce_loss = 0.0
        test_dice_loss = 0.0

        with torch.no_grad():

            for i, sample in enumerate(tbar):
                # print(sample.keys())
                image, target = sample['image'], sample['label']
                # print(image.shape)
                if self.device != 'cpu':
                    image, target = image.cuda(), target.cuda()
                output = self.model(image)

                if (i + 1) % 10 == 0:
                    _output = torch.sigmoid(output).detach().cpu().numpy()
                    _output[_output < 0.5] = 0
                    _output[_output >= 0.5] = 1

                    import matplotlib.pyplot as plt
                    plt.imshow(_output[0, 0, :, :])
                    plt.savefig(f'{self.args.save_dir}/{int(time.time())}_{client_id}_{i}.png')
                    # plt.show()
                    plt.close()

                bce_loss = self.criterion(torch.sigmoid(output), target)
                test_bce_loss += bce_loss.item()
                dice_loss_ = dice_loss(output, target)
                test_dice_loss += dice_loss_

                tbar.set_description('Test loss: %.3f' % (test_bce_loss / (i + 1)))

            self.logger.info(
                f'[{client_id}] Test BCE loss: {test_bce_loss / num_img_tr} Test dice loss: {test_dice_loss / num_img_tr}'
            )

        return {
            'test_bce_loss': test_bce_loss / num_img_tr,
            'test_dice_loss': test_dice_loss / num_img_tr
        }
