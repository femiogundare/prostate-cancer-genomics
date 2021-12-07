# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 11:04:39 2021

@author: femiogundare
"""

import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utilities.plots import plot_roc


def save_coef(fs_model_list, columns, directory, relevant_features):
    
    coef_df = pd.DataFrame(index=columns)
    
    dir_name = os.path.join(directory, 'fs')
    
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    if hasattr(columns, 'levels'):
        genes_list = columns.levels[0]
    else:
        genes_list = columns

    for model, model_params in fs_model_list:
        model_name = get_model_id(model_params)
        c_ = model.get_coef()
        
        logging.info('Saving coef...')

        model_name_col = model_name

        if hasattr(model, 'get_named_coef'):
            print('Save_feature_importance...')
            file_name = os.path.join(dir_name, 'coef_' + model_name)
            coef = model.get_named_coef()
            if type(coef) == list:
                for i, c in enumerate(coef):
                    if type(c) == pd.DataFrame:
                        c.to_csv(file_name + str(i) + '.csv')

        if type(c_) == list:
            coef_df[model_name_col] = c_[0]
        else:
            coef_df[model_name_col] = c_


    if not relevant_features is None:
        for model, model_name in fs_model_list:
            c = model.get_coef()
            if type(c_) == list:
                c = c_[0]
            else:
                c = c_
            plot_roc(relevant_features, c, dir_name, label=model_name)

    plt.savefig(os.path.join(dir_name, 'auc_curves'))
    file_name = os.path.join(dir_name, 'coef.csv')
    coef_df.to_csv(file_name)
    
    
    
def report_density(model_list):
    logging.info('Model density')

    for model, model_params in model_list:
        model_name = get_model_id(model_params)
        logging.info('' + model_name + ': ' + str(model.get_density()))
        
        
        
def get_model_id(model_params):
    if 'id' in model_params:
        model_name = model_params['id']
    else:
        model_name = model_params['type']
        
    return model_name



def get_coef(coef_):
    if coef_.ndim == 1:
        coef = np.abs(coef_)
    else:
        coef = np.sum(np.abs(coef_), axis=0)
        
    return coef



def get_coef_from_model(model):
    coef = None
    if hasattr(model, 'coef_'):
        if type(model.coef_) == list:
            coef = [get_coef(c) for c in model.coef_]
        elif type(model.coef_) == dict:
            coef = [get_coef(model.coef_[c]) for c in model.coef_.keys()]
        else:
            coef = get_coef(model.coef_)

    if hasattr(model, 'scores_'):
        coef = model.scores_

    if hasattr(model, 'feature_importances_'):
        coef = np.abs(model.feature_importances_)
        
    return coef



# get balanced x and y where the size of positive samples equal the number of negative samples
def get_balanced(x, y, info):
    pos_ind = np.where(y == 1.)[0]
    neg_ind = np.where(y == 0.)[0]
    n_pos = pos_ind.shape[0]
    n_neg = neg_ind.shape[0]
    n = min(n_pos, n_neg)

    pos_ind = np.random.choice(pos_ind, size=n, replace=False)
    neg_ind = np.random.choice(neg_ind, size=n, replace=False)

    ind = np.concatenate([pos_ind, neg_ind])
    y = y[ind]
    x = x[ind, :]
    info = info.iloc[ind].copy()
    
    return x, y, info