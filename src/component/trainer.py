#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   experiments\fedeye\src\component\trainer.py
# @Time    :   2022-03-02 17:10:05
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

import wandb
import time
from copy import deepcopy

from tqdm import tqdm, trange

from torchvision.utils import make_grid

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import fedhf
from fedhf.component import BaseTrainer

from .metric import *


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class Trainer(BaseTrainer):

    def __init__(self, args):
        super(Trainer, self).__init__(args)
        self.epoch = 0

    def train_epoch(self):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            # print(sample.keys())
            image, target = sample['image'], sample['label']
            # print(image.shape)
            if self.device != 'cpu':
                image, target = image.cuda(), target.cuda()
            self.optimizer.zero_grad()
            output = self.model(image)
            l2_reg = self._calc_l2_reg(self.model_, self.model)
            loss = self.criterion(torch.sigmoid(output),
                                  target) + l2_reg * self.args.fedasync_rho / 2
            # + dice_loss(output, target)

            # _output = torch.sigmoid(output).detach().cpu().numpy()
            # _output[_output < 0.5] = 0
            # _output[_output >= 0.5] = 1

            # import matplotlib.pyplot as plt
            # plt.imshow(_output[0, 0, :, :])
            # plt.show()

            # print(f'count 0: {torch.sum(_output == 0).item()}')
            # print(f'count 1: {torch.sum(_output == 1).item()}')

            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))

        self.logger.info('[Epoch {}] Train loss: {:.5f}'.format(self.epoch,
                                                                train_loss / num_img_tr))

        self.train_loss = train_loss / num_img_tr

    def train(self, dataloader, model, num_epochs, client_id=None, gpus=[], device='cpu'):
        self.max_epoch = num_epochs
        self.device = device
        self.model = model.to(device)
        self.model_ = deepcopy(model)
        self.model_.to(device)

        self.train_loader = dataloader
        self.optimizer = self.optim(params=self.model.parameters(), lr=self.args.lr)
        self.scheduler = self.lr_scheduler(self.optimizer, self.args.lr_step)
        self.criterion = nn.BCELoss()
        # self.criterion = DiceLoss()

        for epoch in trange(self.epoch, self.max_epoch, desc='Train', ncols=80):
            torch.cuda.empty_cache()
            self.epoch = epoch
            self.train_epoch()
            self.scheduler.step()

        # torch.save(self.model.state_dict(), './chkp/model.pth')

        return {'model': self.model, 'train_loss': self.train_loss}

    def _calc_l2_reg(self, global_model, model):
        l2_reg = 0
        for p1, p2 in zip(global_model.parameters(), model.parameters()):
            l2_reg += (p1 - p2).norm(2)
        return l2_reg