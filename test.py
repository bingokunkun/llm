#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/11/11 00:24
# @Author  : kunkun
# @File    : test.py
# @Project : mixture-of-experts-master
# @Software: PyCharm
import torch.nn as nn
from inspect import isfunction


def a():
    return 0


if __name__ == "__main__":
    print(hasattr(nn, 'GELU'))