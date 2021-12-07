# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 18:35:11 2021

@author: Dell
"""

import copy
import logging
import numpy as np
from sklearn import preprocessing


def get_preprocessor(args):
    print(args)
    preprocessing_type = args['type']
    logging.info('Pre-processing: {}'.format(preprocessing_type))
    
    if preprocessing_type == 'standard':  # 0 mean , 1 variance
        if 'params' in args:
            p1 = args['params']
            preprocessor = preprocessing.StandardScaler(**p1)
        else:
            preprocessor = preprocessing.StandardScaler()
            
    elif preprocessing_type == 'normalize':  # 1 norm
        preprocessor = preprocessing.Normalizer()

    elif preprocessing_type == 'scale':  # 0:1 scale
        if 'params' in args:
            p1 = args['params']
            preprocessor = preprocessing.MinMaxScaler(**p1)
        else:
            preprocessor = preprocessing.MinMaxScaler()
            
    elif preprocessing_type == 'tfidf':
        from sklearn.feature_extraction.text import TfidfTransformer
        p1 = args['params']
        preprocessor = TfidfTransformer(**p1)

    else:
        preprocessor = None

    return preprocessor



def remove_outliers(y):
    m = np.mean(y)
    s = np.std(y)
    y2 = copy.deepcopy(y)
    s = np.std(y)
    n = 4
    y2[y > m + n * s] = m + n * s
    y2[y < m - n * s] = m - n * s
    
    return y2