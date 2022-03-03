#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   experiments\fedeye\src\sampler.py
# @Time    :   2022-03-02 14:36:14
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from dataset import transform as tr

import os
import sys

import numpy as np

import fedhf
from fedhf.core import SimulatedBaseCoordinator, build_client, build_server
from fedhf.component import build_sampler

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


class Coordinator(SimulatedBaseCoordinator):
    """Simulated Coordinator
        In simulated scheme, the data and model belong to coordinator and there is no need communicator.
        Also, there is no need to instantiate every client.
    """
    def __init__(self, args) -> None:
        super(Coordinator, self).__init__(args)

    def prepare(self) -> None:
        self.sampler = build_sampler(self.args.sampler)(self.args)

        self.train_data = self.sampler.sample(split='train', tr=composed_transforms_tr)
        self.test_data = self.sampler.sample(split='test', tr=composed_transforms_ts)

        self.client_list = [i for i in range(self.args.num_clients)]
        self.server = build_server(self.args.deploy_mode)(self.args)

        assert self.args.deploy_mode == 'simulated'

    def main(self) -> None:
        try:
            for i in range(self.args.num_rounds):
                selected_client = self.server.select(self.client_list)

                self.logger.info(f'Round {i} selected client: {selected_client}')

                for client_id in selected_client:
                    model = deepcopy(self.server.model)
                    client = build_client(self.args.deploy_mode)(self.args, client_id)
                    model = client.train(self.train_data[client_id], model)
                    self.server.update(
                        model,
                        server_model_version=self.server.model.get_model_version(),
                        client_id=client_id)

                result = self.server.evaluate(self.train_data)
                self.logger.info(f'Server result: {result}')

                if self.server.model.get_model_version() % self.args.check_point == 0:
                    self.server.model.save(
                        f'{self.args.name}-{self.server.model.get_model_version()}.pth')

            self.logger.info(f'All rounds finished.')

        except KeyboardInterrupt:
            self.server.model.save()
            self.logger.info(f'Interrupted by user.')

    def finish(self) -> None:
        super(Coordinator, self).finish()

    def run(self) -> None:
        self.prepare()
        self.main()
        self.finish()
