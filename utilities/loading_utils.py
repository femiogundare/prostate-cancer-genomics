# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 01:43:53 2021

@author: femiogundare
"""

import os
import yaml
import numpy as np
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from data_scripts.data_access import Data
from model import model_factory


class DataModelLoader:
    def __init__(self, params_file):
        self.dir_path = os.path.dirname(os.path.realpath(params_file))
        model_parmas, data_parmas = self.load_params(params_file)
        data_reader = Data(**data_parmas)
        self.model = None
        X_train, X_val, X_test, y_train, y_val, y_test, samples_train, samples_val, samples_test, genes = data_reader.get_train_val_test()

        self.X_train = X_train
        self.X_test = np.concatenate([X_val, X_test], axis=0)

        self.y_train = y_train
        self.y_test = np.concatenate([y_val, y_test], axis=0)

        self.samples_train = samples_train
        self.samples_test = list(samples_val) + list(samples_test)
        
        self.genes = genes
        
    def get_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test, self.samples_train, self.samples_test, self.genes
    
    def get_model(self, model_name='P-net_params.yml'):
        self.model = self.load_model(self.dir_path, model_name)
        return self.model
    
    def load_model(self, model_dir_, model_name):
        # 1 - load architecture
        params_filename = os.path.join(model_dir_, model_name + '_params.yml')
        stream = open(params_filename, 'r')
        params = yaml.load(stream)
        print('Params', params)
        fs_model = model_factory.get_model(params['model_params'])
        #print(fs_model.summary())
        print('FS model', fs_model)
        # 2 -compile model and load weights (link weights)
        weights_file = os.path.join(model_dir_, 'fs/{}.h5'.format(model_name))
        model = fs_model.load_model(weights_file)
        return model
    
    def load_params(self, params_filename):
        stream = open(params_filename, 'r')
        params = yaml.load(stream, Loader=yaml.UnsafeLoader)
        model_params = params['model_params']
        data_params = params['data_params']
        return model_params, data_params