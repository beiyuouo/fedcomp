#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   experiments\fedeye\src\main_fedasync.py
# @Time    :   2022-03-22 18:34:39
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
from component.coordinator import SyncCoordinator, AsyncCoordinator, AsyncCoordinatorUnlimited
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
    '0.25',
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
    '12',
    '--num_local_epochs',
    '3',
    '--sampler',
    'fundus_sampler',
    '--model',
    'unet_mini',
    '--unet_n1',
    '4',
    '--unet_bilinear',
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
    '--seed',
    '2333',
    '--select_ratio',
    '0.25',
    '--evaluation_interval',
    '1',
    '--evaluate_on_client',
])

Injector.register('model', {'deeplab': DeepLab})
Injector.register('dataset', {'fundus': FundusSegmentation})
Injector.register('sampler', {'fundus_sampler': FundusSampler})
Injector.register('trainer', {'fundus_trainer': Trainer})
Injector.register('evaluator', {'fundus_evaluator': Evaluator})


def main():
    coo = AsyncCoordinator(args)
    # coo = SyncCoordinator(args)
    coo.run()


def main_unlimited():
    coo = AsyncCoordinatorUnlimited(args)
    coo.run()


if __name__ == '__main__':
    main()
    # main_unlimited()