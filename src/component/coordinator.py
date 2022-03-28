#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   experiments\fedeye\src\sampler.py
# @Time    :   2022-03-02 14:36:14
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

from copy import deepcopy
from importlib.resources import path

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
from fedhf.model import build_model
from fedhf.api.serial import Serializer

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


class SyncCoordinator(SimulatedBaseCoordinator):
    """Simulated Coordinator
        In simulated scheme, the data and model belong to coordinator and there is no need communicator.
        Also, there is no need to instantiate every client.
    """

    def __init__(self, args) -> None:
        super(SyncCoordinator, self).__init__(args)

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
                    model = client.train(self.train_data[client_id], model, device=self.args.device)
                    self.server.update(model,
                                       server_model_version=self.server.model.get_model_version(),
                                       client_id=client_id)

                if i % self.args.evaluation_interval == 0:
                    self.logger.info(f'Round {i} evaluate on server')
                    for client_id in self.client_list:
                        result = self.server.evaluate(self.test_data[client_id])
                        self.logger.info(f'Test result on Client {client_id}: {result}')

                if self.server.model.get_model_version() % self.args.checkpoint_interval == 0:
                    self.server.model.save(
                        f'{self.args.name}-{self.server.model.get_model_version()}.pth')

            self.logger.info(f'All rounds finished.')

        except KeyboardInterrupt:
            self.server.model.save()
            self.logger.info(f'Interrupted by user.')

    def finish(self) -> None:
        self.server.model.save()

        try:
            if self.args.evaluate_on_client:
                self.logger.info("Evaluate on client")
                for client_id in self.client_list:
                    client = build_client(self.args.deploy_mode)(self.args, client_id)
                    result = client.evaluate(data=self.train_data[client_id],
                                             model=self.server.model)
                    self.logger.info(f'Train result on Client {client_id}: {result}')

            for client_id in self.client_list:
                result = self.server.evaluate(self.test_data[client_id])
                self.logger.info(f'Test result on Client {client_id}: {result}')

            self.logger.info(f'Final server model version: {self.server.model.get_model_version()}')
        except KeyboardInterrupt:
            self.logger.info(f'Interrupted by user.')

        self.logger.info(f'All finished.')

    def run(self) -> None:
        self.prepare()
        self.main()
        self.finish()


class AsyncCoordinator(SimulatedBaseCoordinator):

    def __init__(self, args) -> None:
        super(AsyncCoordinator, self).__init__(args)

    def prepare(self) -> None:
        self.sampler = build_sampler(self.args.sampler)(self.args)

        self.train_data = self.sampler.sample(split='train', tr=composed_transforms_tr)
        self.test_data = self.sampler.sample(split='test', tr=composed_transforms_ts)

        self.client_list = [i for i in range(self.args.num_clients)]
        self.server = build_server(self.args.deploy_mode)(self.args)

        assert self.args.deploy_mode == 'simulated'

        self._model_queue = []
        self._last_update_time = {client_id: 0 for client_id in self.client_list}

    def main(self) -> None:
        try:
            self._model_queue.append(deepcopy(self.server.model))

            for i in range(self.args.num_rounds):
                # self.logger.info(f'{self.server.model.get_model_version()}')

                selected_clients = self.server.select(self.client_list)

                self.logger.info(f'Round {i} Selected clients: {selected_clients}')

                for client_id in selected_clients:
                    client = build_client(self.args.deploy_mode)(self.args, client_id)

                    staleness = np.random.randint(
                        low=1,
                        high=min(
                            self.args.fedasync_max_staleness,
                            max(self.server.model.get_model_version(), 0) + 1,
                            self.server.model.get_model_version() -
                            self._last_update_time[client_id] + 1) + 1)

                    assert staleness <= max(0, self.server.model.get_model_version()) + 1
                    assert staleness <= len(self._model_queue)
                    assert staleness <= self.server.model.get_model_version(
                    ) - self._last_update_time[client_id] + 1

                    self.logger.info(
                        f'Client {client_id} staleness: {staleness} start train from model version : {self._model_queue[-staleness].get_model_version()}'
                    )

                    model = client.train(data=self.train_data[client_id],
                                         model=deepcopy(self._model_queue[-staleness]),
                                         device=self.args.device)

                    self.server.update(model,
                                       server_model_version=self.server.model.get_model_version(),
                                       client_model_version=model.get_model_version())

                    if self.server.model.get_model_version() % self.args.checkpoint_interval == 0:
                        self.logger.info(
                            f'Save model: {self.args.name}-{self.server.model.get_model_version()}.pth'
                        )
                        self.server.model.save(
                            os.path.join(
                                self.args.save_dir,
                                f'{self.args.name}-{self.server.model.get_model_version()}.pth'))

                    if self.server.model.get_model_version() % self.args.evaluation_interval == 0:
                        self.logger.info(f'Round {i} evaluate on server')
                        for client_id in self.client_list:
                            result = self.server.evaluate(self.train_data[client_id])
                            self.logger.info(f'Train result on Client {client_id}: {result}')

                            result = self.server.evaluate(self.test_data[client_id])
                            self.logger.info(f'Test result on Client {client_id}: {result}')

                    if self.server.model.get_model_version() % 10 == 0:
                        self.args.lr = self.args.lr * 0.25

                    self._model_queue.append(deepcopy(self.server.model))
                    self._last_update_time[client_id] = self.server.model.get_model_version()

                    while len(self._model_queue) > self.args.fedasync_max_staleness + 1:
                        self._model_queue.pop(0)

            self.logger.info(f'All rounds finished.')

        except KeyboardInterrupt:
            self.server.model.save()
            self.logger.info(f'Interrupted by user.')

    def finish(self) -> None:
        self.server.model.save()
        self.server.model.save('model.pth')

        try:
            if self.args.evaluate_on_client:
                self.logger.info("Evaluate on client")
                for client_id in self.client_list:
                    client = build_client(self.args.deploy_mode)(self.args, client_id)
                    result = client.evaluate(data=self.train_data[client_id],
                                             model=self.server.model)
                    self.logger.info(f'Train result on Client {client_id}: {result}')

            for client_id in self.client_list:
                result = self.server.evaluate(self.train_data[client_id])
                self.logger.info(f'Train result on Client {client_id}: {result}')

                result = self.server.evaluate(self.test_data[client_id])
                self.logger.info(f'Test result on Client {client_id}: {result}')

            self.logger.info(f'Final server model version: {self.server.model.get_model_version()}')
        except KeyboardInterrupt:
            self.logger.info(f'Interrupted by user.')

        self.logger.info(f'All finished.')

    def run(self) -> None:
        self.prepare()
        self.main()
        self.finish()


class FedEyeCoordinator(SimulatedBaseCoordinator):

    def __init__(self, args) -> None:
        super(FedEyeCoordinator, self).__init__(args)

    def prepare(self) -> None:
        self.sampler = build_sampler(self.args.sampler)(self.args)

        self.train_data = self.sampler.sample(split='train', tr=composed_transforms_tr)
        self.test_data = self.sampler.sample(split='test', tr=composed_transforms_ts)

        self.client_list = [i for i in range(self.args.num_clients)]
        self.server = build_server(self.args.deploy_mode)(self.args)

        assert self.args.deploy_mode == 'simulated'

        self._update_order = np.random.choice(self.client_list, self.args.num_rounds)
        self.logger.info(f'update order: {self._update_order}')
        self._last_update_time = {i: 0 for i in self.client_list}

        self._temp_dir = './temp'
        if not os.path.exists(self._temp_dir):
            os.makedirs(self._temp_dir)

        self._model_path = []

    def main(self) -> None:
        try:
            self.server.model.save(path=os.path.join(self._temp_dir, f'model_{0}.pth'))
            self._model_path.append(os.path.join(self._temp_dir, f'model_{0}.pth'))

            for i in range(self.args.num_rounds):
                # self.logger.info(f'{self.server.model.get_model_version()}')

                client_id = self._update_order[i]

                client = build_client(self.args.deploy_mode)(self.args, client_id)

                self.logger.info(
                    f'Round {i} start train on Client {client_id} with model version : {self._last_update_time[client_id]}'
                )

                _model = build_model(self.args.model)(self.args)
                _model.load(path=self._model_path[self._last_update_time[client_id]])

                model = client.train(data=self.train_data[client_id],
                                     model=deepcopy(_model),
                                     device=self.args.device)

                client_grad = torch.sub(Serializer.serialize_model(model),
                                        Serializer.serialize_model(_model))

                self.server.update(model,
                                   server_model_version=self.server.model.get_model_version(),
                                   client_id=client_id,
                                   client_grad=client_grad,
                                   model_structure=self.server.model)

                if self.server.model.get_model_version() % self.args.checkpoint_interval == 0:
                    self.logger.info(
                        f'Save model: {self.args.name}-{self.server.model.get_model_version()}.pth')
                    self.server.model.save(
                        os.path.join(
                            self.args.save_dir,
                            f'{self.args.name}-{self.server.model.get_model_version()}.pth'))

                if self.server.model.get_model_version() % self.args.evaluation_interval == 0:
                    self.logger.info(f'Round {i} evaluate on server')
                    for client_id in self.client_list:
                        result = self.server.evaluate(self.train_data[client_id])
                        self.logger.info(f'Train result on Client {client_id}: {result}')

                        result = self.server.evaluate(self.test_data[client_id])
                        self.logger.info(f'Test result on Client {client_id}: {result}')

                if self.server.model.get_model_version() % 10 == 0:
                    self.args.lr = self.args.lr * 0.25

                self.server.model.save(path=os.path.join(
                    self._temp_dir, f'model_{self.server.model.get_model_version()}.pth'))
                self._model_path.append(
                    os.path.join(self._temp_dir,
                                 f'model_{self.server.model.get_model_version()}.pth'))
                self._last_update_time[client_id] = self.server.model.get_model_version()

            self.logger.info(f'All rounds finished.')

        except KeyboardInterrupt:
            self.server.model.save()
            self.logger.info(f'Interrupted by user.')

    def finish(self) -> None:
        self.server.model.save()

        try:
            if self.args.evaluate_on_client:
                self.logger.info("Evaluate on client")
                for client_id in self.client_list:
                    client = build_client(self.args.deploy_mode)(self.args, client_id)
                    result = client.evaluate(data=self.train_data[client_id],
                                             model=self.server.model)
                    self.logger.info(f'Train result on Client {client_id}: {result}')

            for client_id in self.client_list:
                result = self.server.evaluate(self.train_data[client_id])
                self.logger.info(f'Train result on Client {client_id}: {result}')

                result = self.server.evaluate(self.test_data[client_id])
                self.logger.info(f'Test result on Client {client_id}: {result}')

            self.logger.info(f'Final server model version: {self.server.model.get_model_version()}')
        except KeyboardInterrupt:
            self.logger.info(f'Interrupted by user.')

        self.logger.info(f'All finished.')


class AsyncCoordinatorUnlimited(SimulatedBaseCoordinator):

    def __init__(self, args) -> None:
        super(AsyncCoordinatorUnlimited, self).__init__(args)

    def prepare(self) -> None:
        self.sampler = build_sampler(self.args.sampler)(self.args)

        self.train_data = self.sampler.sample(split='train', tr=composed_transforms_tr)
        self.test_data = self.sampler.sample(split='test', tr=composed_transforms_ts)

        self.client_list = [i for i in range(self.args.num_clients)]
        self.server = build_server(self.args.deploy_mode)(self.args)

        assert self.args.deploy_mode == 'simulated'

        self._update_order = np.random.choice(self.client_list, self.args.num_rounds)
        self.logger.info(f'update order: {self._update_order}')
        self._last_update_time = {i: 0 for i in self.client_list}

        self._temp_dir = './temp'
        if not os.path.exists(self._temp_dir):
            os.makedirs(self._temp_dir)

        self._model_path = []

    def main(self) -> None:
        try:
            self.server.model.save(path=os.path.join(self._temp_dir, f'model_{0}.pth'))
            self._model_path.append(os.path.join(self._temp_dir, f'model_{0}.pth'))

            for i in range(self.args.num_rounds):
                # self.logger.info(f'{self.server.model.get_model_version()}')

                client_id = self._update_order[i]

                client = build_client(self.args.deploy_mode)(self.args, client_id)

                self.logger.info(
                    f'Round {i} start train on Client {client_id} with model version : {self._last_update_time[client_id]}'
                )

                _model = build_model(self.args.model)(self.args)
                _model.load(path=self._model_path[self._last_update_time[client_id]])

                model = client.train(data=self.train_data[client_id],
                                     model=deepcopy(_model),
                                     device=self.args.device)

                self.server.update(model,
                                   server_model_version=self.server.model.get_model_version(),
                                   client_model_version=model.get_model_version())

                if self.server.model.get_model_version() % self.args.checkpoint_interval == 0:
                    self.logger.info(
                        f'Save model: {self.args.name}-{self.server.model.get_model_version()}.pth')
                    self.server.model.save(
                        os.path.join(
                            self.args.save_dir,
                            f'{self.args.name}-{self.server.model.get_model_version()}.pth'))

                if self.server.model.get_model_version() % self.args.evaluation_interval == 0:
                    self.logger.info(f'Round {i} evaluate on server')
                    for client_id in self.client_list:
                        result = self.server.evaluate(self.train_data[client_id])
                        self.logger.info(f'Train result on Client {client_id}: {result}')

                        result = self.server.evaluate(self.test_data[client_id])
                        self.logger.info(f'Test result on Client {client_id}: {result}')

                if self.server.model.get_model_version() % 10 == 0:
                    self.args.lr = self.args.lr * 0.25

                self.server.model.save(path=os.path.join(
                    self._temp_dir, f'model_{self.server.model.get_model_version()}.pth'))
                self._model_path.append(
                    os.path.join(self._temp_dir,
                                 f'model_{self.server.model.get_model_version()}.pth'))
                self._last_update_time[client_id] = self.server.model.get_model_version()

            self.logger.info(f'All rounds finished.')

        except KeyboardInterrupt:
            self.server.model.save()
            self.logger.info(f'Interrupted by user.')

    def finish(self) -> None:
        self.server.model.save()

        try:
            if self.args.evaluate_on_client:
                self.logger.info("Evaluate on client")
                for client_id in self.client_list:
                    client = build_client(self.args.deploy_mode)(self.args, client_id)
                    result = client.evaluate(data=self.train_data[client_id],
                                             model=self.server.model)
                    self.logger.info(f'Train result on Client {client_id}: {result}')

            for client_id in self.client_list:
                result = self.server.evaluate(self.train_data[client_id])
                self.logger.info(f'Train result on Client {client_id}: {result}')

                result = self.server.evaluate(self.test_data[client_id])
                self.logger.info(f'Test result on Client {client_id}: {result}')

            self.logger.info(f'Final server model version: {self.server.model.get_model_version()}')
        except KeyboardInterrupt:
            self.logger.info(f'Interrupted by user.')

        self.logger.info(f'All finished.')
