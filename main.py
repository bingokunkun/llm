#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/11/9 23:43
# @Author  : kunkun
# @File    : main.py
# @Project : mixture-of-experts-master
# @Software: PyCharm
import torch
from torch import nn
from mixture_of_experts import MoE


moe = MoE(dim=512, num_experts=16, hidden_dim=512 * 4, activation=nn.LeakyReLU)
inputs = torch.randn(4, 1024, 512)
out, aux_loss = moe(inputs)  # (4, 1024, 512), (1,)
print(out.shape)
print(aux_loss)