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

from .metric import BinaryDiceLoss, DiceLoss


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
            if self.device is not 'cpu':
                image, target = image.cuda(), target.cuda()
            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(torch.sigmoid(output), target)
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

        self.train_loader = dataloader
        self.optimizer = self.optim(params=model.parameters(), lr=self.args.lr)
        self.scheduler = self.lr_scheduler(self.optimizer, self.args.lr_step)
        # self.criterion = nn.BCELoss()
        self.criterion = BinaryDiceLoss()

        for epoch in trange(self.epoch, self.max_epoch, desc='Train', ncols=80):
            torch.cuda.empty_cache()
            self.epoch = epoch
            self.train_epoch()
            self.scheduler.step()

        return {'model': self.model, 'train_loss': self.train_loss}