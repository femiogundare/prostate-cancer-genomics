# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 11:21:23 2021

@author: femiogundare
"""

from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[4]))

from model.builders.prostate_models import build_pnet

task = 'classification_binary'
selected_genes = 'tcga_prostate_expressed_genes_and_cancer_genes.csv'

data_base = {'id': 'ALL', 'type': 'prostate_data',
             'params': {
                 'data_type': ['important_mutations', 'cnv'],
                 'drop_AR': False,
                 'cnv_levels': 3,
                 'mutation_binary': True,
                 'balanced_data': False,
                 'combine_type': 'union',  # intersection
                 'use_coding_genes_only': True,
                 'selected_genes': selected_genes,
                 'training_split': 0,
                 }
             }

data = [data_base]

n_hidden_layers = 5
base_dropout = 0.5
wregs = [0.001] * 7
loss_weights = [2, 7, 20, 54, 148, 400]
wreg_outcomes = [0.01] * 6
pre = {'type': None}


nn_pathway = {
    'type': 'nn',
    
    'id': 'P-net',
    
    'params':
        {
            'build_fn': build_pnet,
            
            'model_params': {
                'use_bias': True,
                'w_reg': wregs,
                'w_reg_outcomes': wreg_outcomes,
                'dropout': [base_dropout] + [0.1] * (n_hidden_layers + 1),
                'loss_weights': loss_weights,
                'optimizer': 'Adam',
                'activation': 'tanh',
                'data_params': data_base,
                'add_unk_genes': False,
                'shuffle_genes': False,
                'kernel_initializer': 'lecun_uniform',
                'n_hidden_layers': n_hidden_layers,
                'attention': False,
                'dropout_testing': False  # keep dropout in testing phase, useful for bayesian inference
                }, 
            
            'fitting_params': dict(samples_per_epoch=10,
                                      select_best_model=False,
                                      monitor='val_o6_f1',
                                      verbose=2,
                                      epoch=35,
                                      shuffle=True,
                                      batch_size=50,
                                      save_name='pnet',
                                      debug=False,
                                      save_gradient=False,
                                      class_weight='auto',
                                      n_outputs=n_hidden_layers + 1,
                                      prediction_output='average',
                                      early_stop=False,
                                      reduce_lr=False,
                                      reduce_lr_after_nepochs=dict(drop=0.25, epochs_drop=10),
                                      lr=0.001,
                                      max_f1=True
                                      ),
            'feature_importance': 'deepexplain_deeplift'
            }
        }
    
features = {}
models = [nn_pathway]

pipeline = {'type': 'Train_Validate', 'params': {'save_train': True}}