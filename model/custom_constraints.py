# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 08:26:34 2021

@author: femiogundare
"""

import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.constraints import Constraint

"""
import os
import random
import tensorflow as tf
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

random_seed = 234
os.environ['PYTHONHASHSEED'] = str(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)
tf.random.set_seed(random_seed)
tf.compat.v1.set_random_seed(random_seed)
session_config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
session = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_config)
tf.compat.v1.keras.backend.set_session(session)
"""


class OneToOne(Constraint):
    def __call__(self, p):
        p *= K.identity(p)
        return p
    

class ConnectionConstraints(Constraint):
    def __init__(self, m):
        self.connection_map = m
        print('Building a kernel constraint')
        
    def __call__(self, p):
        mapp = np.array(self.connection_map)
        p *= mapp.astype(K.floatx())
        return p
    
    def get_config(self):
        return {
            'name': self.__class__.__name__,
            'map': self.connection_map
        }
    
    
class ZeroWeights(Constraint):
    def __call__(self, p):
        p = K.zeros_like(p)
        return p
    

class OneWeights(Constraint):
    def __call__(self, p):
        p = K.ones_like(p)
        return p