#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   experiments\fedeye\src\component\aggregator.py
# @Time    :   2022-03-22 16:31:07
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from fedhf.component import AsyncAggregator


class FedEyeAggregator(AsyncAggregator):

    def __init__(self, args) -> None:
        super(FedEyeAggregator, self).__init__(args)

        self._update_time = {}
        self._total_client = 0
        self._cur_round = 0

    def agg(self, server_param: torch.Tensor, client_param: torch.Tensor, **kwargs):
        if not self._check_agg():
            return

        # check parameter
        for param in ['client_id', 'client_grad', 'server_model_version']:
            if param not in kwargs:
                raise ValueError(f'{param} is required')

        client_id = kwargs['client_id']
        client_grad = kwargs['client_grad']
        server_model_version = kwargs['server_model_version']
        self._cur_round += 1

        if client_id not in self._update_time:
            # solve initial error
            self._total_client += 1
            _expect_times = max(self._cur_round / self.args.num_clients, 1)
            new_param = server_param * (
                (self._total_client + 1) / self._total_client) + client_param * (
                    1 / self._total_client) * (1 / (np.log(_expect_times) + 1))

        else:
            staleness = server_model_version - self._update_time[client_id] + 1
            update_from_param = torch.sub(client_param, client_grad)
            gc = torch.sub(client_param, update_from_param)
            gs = torch.sub(server_param, update_from_param) * (np.log(staleness) + 1)
            alpha = torch.div(torch.dot(gs, gc), gc.norm()**2)

            if not torch.equal(
                    gs, torch.zeros_like(gs)) and torch.is_nonzero(alpha) and torch.dot(gc, gs) < 0:
                self.logger.info(f"alpha is {alpha}")
                new_param = torch.add(update_from_param, torch.sub(gs, torch.mul(alpha, gc)))
            else:
                self.logger.info(f"add gc directly")
                new_param = server_param + self.alpha * gc

        self._update_time[client_id] = server_model_version + 1

        result = {
            'param': new_param,
            'model_version': server_model_version + 1,
            'model_time': time.time()
        }
        return result