#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   experiments\fedeye\src\sampler.py
# @Time    :   2022-03-02 14:36:14
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys

import numpy as np

import fedhf
from fedhf.component import BaseSampler
from fedhf.dataset import ClientDataset
from dataset.fundus import FundusSegmentation


class FundusSampler(BaseSampler):
    def __init__(self, args):
        super(FundusSampler, self).__init__(args)

    def sample(self, split: str = 'train', tr=None):
        datasets = []
        for i in range(1, 5):
            dataset_path = f'Domain{i}'
            dataset = FundusSegmentation(base_dir=self.args.data_dir,
                                         dataset=dataset_path,
                                         split=split,
                                         transform=tr)
            datasets.append(dataset)

        return datasets
