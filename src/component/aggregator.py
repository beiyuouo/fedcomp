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

        # self._initialization_error = None

        self._gamma = 0.5

    def agg(self, server_param: torch.Tensor, client_param: torch.Tensor, **kwargs):
        if not self._check_agg():
            return

        # check parameter
        for param in ['client_id', 'client_grad', 'server_model_version', 'model_structure']:
            if param not in kwargs:
                raise ValueError(f'{param} is required')

        client_id = kwargs['client_id']
        client_grad = kwargs['client_grad']
        server_model_version = kwargs['server_model_version']
        model = kwargs['model_structure']
        self._cur_round += 1

        new_param = None

        if client_id not in self._update_time.keys():
            # solve initial error
            self._total_client += 1
            if self._cur_round == 1:
                new_param = client_param
                # self._initialization_error = torch.sub(new_param, server_param)
            else:
                staleness = server_model_version - self._update_time[client_id] + 1
                staleness = torch.tensor(staleness, dtype=torch.float32)
                new_param = self._solve_conflict(staleness, server_param, client_param, client_grad,
                                                 model)

        else:
            staleness = server_model_version - self._update_time[client_id] + 1
            staleness = torch.tensor(staleness, dtype=torch.float32)

            new_param = self._solve_conflict(staleness, server_param, client_param, client_grad,
                                             model)

        self._update_time[client_id] = server_model_version + 1

        result = {
            'param': new_param,
            'model_version': server_model_version + 1,
            'model_time': time.time()
        }
        return result

    def _solve_initialization_error(self):
        pass

    def _solve_conflict(self, staleness, server_param, client_param, client_grad, model):
        cur_idx = 0
        new_param = torch.zeros_like(server_param)
        for parameter in model.parameters():
            numel = parameter.data.numel()
            # size = parameter.data.size()

            _server_param = server_param[cur_idx:cur_idx + numel]
            _client_param = client_param[cur_idx:cur_idx + numel]
            _client_grad = client_grad[cur_idx:cur_idx + numel]

            _update_from_param = torch.sub(_client_param, _client_grad)
            _gc = torch.sub(_client_param, _update_from_param)
            _gs = torch.sub(_server_param, _update_from_param)
            _alpha = torch.div(torch.dot(_gs, _gc),
                               _gc.norm()**2) * ((torch.log(staleness) + 1) * self._gamma)
            self.logger.info(f"alpha is {_alpha}")

            if not torch.equal(_gs, torch.zeros_like(_gs)) and torch.is_nonzero(
                    _alpha) and torch.dot(_gc, _gs) < 0:
                self.logger.info(f"solve conflicts")
                _new_param = torch.add(_update_from_param, torch.sub(_gs, torch.mul(_alpha, _gc)))
            else:
                self.logger.info(f"add gc directly")
                _new_param = torch.add(_server_param, torch.mul(_alpha, _gc))
                # new_param = server_param + self.alpha * gc

            new_param[cur_idx:cur_idx + numel] = _new_param
            cur_idx += numel
        return new_param
