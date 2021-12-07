# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 12:37:52 2021

@author: femiogundare
"""

import os
import imp
import random
import logging
import timeit
import datetime
import numpy as np
import tensorflow as tf
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline.train_validate import TrainValidatePipeline
from pipeline.onesplit_pipeline import OneSplitPipeline
from pipeline.leaveoneout_pipeline import LeaveOneOutPipeline
from pipeline.crossvalidation_pipeline import CrossvalidationPipeline
from config import PROSTATE_LOG_PATH, PROSTATE_PARAMS_PATH
from utilities.logs import set_logging, DebugFolder

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

#os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES'] = ''

random_seed = 234
os.environ['PYTHONHASHSEED'] = str(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)
tf.random.set_seed(random_seed)
tf.compat.v1.set_random_seed(random_seed)
session_config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
session = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_config)
tf.compat.v1.keras.backend.set_session(session)

#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

timeStamp = '_{0:%b}-{0:%d}_{0:%H}-{0:%M}'.format(datetime.datetime.now())


def elapsed_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


params_file_list = []

# Prostate Net
params_file_list.append('prostate_net/onesplit_average_reg_10_tanh_test')
params_file_list.append('prostate_net/onesplit_average_reg_10_tanh_test_2')
params_file_list.append('prostate_net/crossvalidation_average_reg_10_tanh')

# Other ML models
params_file_list.append('other_ml_models/onesplit_ml_test')
params_file_list.append('other_ml_models/crossvalidation_ml')

# Dense Net
params_file_list.append('dense_net/onesplit_dense_pnet_test')
params_file_list.append('dense_net/onesplit_dense_test')

# Number samples
params_file_list.append('number_samples/crossvalidation_average_reg_10')
params_file_list.append('number_samples/crossvalidation_average_reg_10_tanh')
params_file_list.append('number_samples/crossvalidation_number_samples_densenet_sameweights')

# External validation
#params_file_list.append('external_validation/prostate_net_validation')

# Reviews
## Fusions
params_file_list.append('review/fusion/onesplit_average_reg_10_tanh_test_fusion')
params_file_list.append('review/fusion/onesplit_average_reg_10_tanh_test_fusion_zero')
params_file_list.append('review/fusion/onesplit_average_reg_10_tanh_test_inner_fusion_genes')
params_file_list.append('review/fusion/onesplit_average_reg_10_tanh_test_tmb')

## Single copy
#params_file_list.append('review/single_copy/crossvalidation_average_reg_10_tanh_single_copy')
#params_file_list.append('review/single_copy/onesplit_average_reg_10_tanh_test_single_copy')



for params_file in params_file_list:
    log_dir = os.path.join(PROSTATE_LOG_PATH, params_file)
    set_logging(log_dir)
    params_file = os.path.join(PROSTATE_PARAMS_PATH, params_file)
    logging.info('Random seed: {}'.format(random_seed))
    params_file_full = params_file + '.py'
    print(params_file)
    params = imp.load_source(params_file, params_file_full)
    print(params)
    
    DebugFolder(log_dir)
    
    if params.pipeline['type'] == 'one_split':
        pipeline = OneSplitPipeline(task=params.task, data_params=params.data, model_params=params.models,
                                    pre_params=params.pre, feature_params=params.features,
                                    pipeline_params=params.pipeline,
                                    exp_name=log_dir)

    elif params.pipeline['type'] == 'crossvalidation':
        pipeline = CrossvalidationPipeline(task=params.task, data_params=params.data, feature_params=params.features,
                                           model_params=params.models, pre_params=params.pre,
                                           pipeline_params=params.pipeline, exp_name=log_dir)
        
    elif params.pipeline['type'] == 'Train_Validate':
        pipeline = TrainValidatePipeline(data_params=params.data, model_params=params.models, pre_params=params.pre,
                                         feature_params=params.features, pipeline_params=params.pipeline,
                                         exp_name=log_dir)

    elif params.pipeline['type'] == 'LOOCV':
        pipeline = LeaveOneOutPipeline(task=params.task, data_params=params.data, feature_params=params.features,
                                       model_params=params.models, pre_params=params.pre,
                                       pipeline_params=params.pipeline, exp_name=log_dir)
        
        
    start = timeit.default_timer()
    pipeline.run()
    stop = timeit.default_timer()
    mins, secs = elapsed_time(start, stop)
    logging.info('Elapsed Time: {}m {}s'.format(mins, secs))