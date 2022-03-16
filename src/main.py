#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   eyes\src\main.py
# @Time    :   2022-02-21 14:57:19
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

import os
import sys
from time import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torchvision import transforms

import fedhf
from fedhf.core import Injector
from fedhf.api import opts

from dataset.fundus import FundusSegmentation
from component.sampler import FundusSampler
from component.coordinator import SyncCoordinator, AsyncCoordinator
from component.trainer import Trainer
from component.evaluator import Evaluator

from net import DeepLab

args = opts().parse([
    # '--use_wandb',
    '--agg',
    'fedasync',
    '--fedasync_rho',
    '0.005',
    '--fedasync_strategy',
    'polynomial',
    '--fedasync_alpha',
    '0.9',
    '--fedasync_max_staleness',
    '4',
    '--fedasync_a',
    '0.5',
    '--fedasync_b',
    '4',
    '--data_dir',
    os.path.join('..', 'data', 'fundus'),
    '--batch_size',
    '4',
    '--num_clients',
    '4',
    '--num_rounds',
    '1',
    '--num_local_epochs',
    '100',
    '--sampler',
    'fundus_sampler',
    '--model',
    'unet',
    '--unet_n1',
    '16',
    '--lr',
    '0.01',
    '--num_classes',
    '2',
    '--input_c',
    '3',
    '--output_c',
    '2',
    '--trainer',
    'fundus_trainer',
    '--evaluator',
    'fundus_evaluator',
    '--dataset',
    'fundus',
    # '--seed',
    # '1',
    '--select_ratio',
    '0.25',
])

Injector.register('model', {'deeplab': DeepLab})
Injector.register('dataset', {'fundus': FundusSegmentation})
Injector.register('sampler', {'fundus_sampler': FundusSampler})
Injector.register('coordinator', {'fundus_fedavg': SyncCoordinator})
Injector.register('coordinator', {'fundus_fedasync': AsyncCoordinator})
Injector.register('trainer', {'fundus_trainer': Trainer})
Injector.register('evaluator', {'fundus_evaluator': Evaluator})


def main():
    coo = AsyncCoordinator(args)
    coo.run()


if __name__ == '__main__':
    main()