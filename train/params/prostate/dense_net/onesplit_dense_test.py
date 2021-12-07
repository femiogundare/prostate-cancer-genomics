# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 11:01:02 2021

@author: femiogundare
"""

import numpy as np
from copy import deepcopy
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[4]))

from model.builders.prostate_models import build_dense

task = 'classification_binary'
selected_genes = 'tcga_prostate_expressed_genes_and_cancer_genes.csv'

data_base = {'id': 'ALL', 'type': 'prostate_data',
             'params': {
                 'data_type': ['important_mutations', 'cnv_deletion', 'cnv_amplification'],
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

data = []

splits = np.arange(0, 1)

for n in splits:
    d = deepcopy(data_base)
    d['id'] = 'data_{}'.format(n)
    d['params']['training_split'] = str(n)
    data.append(d)

pre = {'type': None}

features = {}


nn_pathway_dense = {
    'type': 'nn',
    
    'id': 'dense',
    
    'params':
        {
            'build_fn': build_dense,
            
            'model_params': {
                'w_reg': 0.01,
                'n_weights': 71009,
                'optimizer': 'Adam',
                'activation': 'selu',
                'data_params': data_base,
                }, 
            
            'fitting_params': dict(samples_per_epoch=10,
                                      select_best_model=False,
                                      monitor='val_f1',
                                      verbose=2,
                                      epoch=300,
                                      shuffle=True,
                                      batch_size=50,
                                      save_name='dense',
                                      debug=False,
                                      save_gradient=False,
                                      class_weight='auto',
                                      n_outputs=1,
                                      prediction_output='average',
                                      early_stop=False,
                                      reduce_lr=False,
                                      reduce_lr_after_nepochs=dict(drop=0.25, epochs_drop=50),
                                      ),
            # 'feature_importance': 'deepexplain_grad*input'
            }
        }

models = [nn_pathway_dense]

pipeline = {'type': 'one_split', 'params': {'save_train': True, 'eval_dataset': 'test'}}