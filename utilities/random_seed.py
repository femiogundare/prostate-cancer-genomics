# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 19:05:51 2021

@author: femiogundare
"""

import numpy as np
import random


def set_random_seeds(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)