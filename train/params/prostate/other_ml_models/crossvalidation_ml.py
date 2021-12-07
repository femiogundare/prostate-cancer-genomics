# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 09:19:45 2021

@author: femiogundare
"""

import os
import copy
import numpy as np

task = 'classification_binary'
selected_genes = 'tcga_prostate_expressed_genes_and_cancer_genes.csv'

data_base = {'id': 'ALL', 'type': 'prostate_data',
             'params': {
                 'data_type': ['important_mutations', 'cnv_deletion', 'cnv_amplification'],
                 'drop_AR': False,
                 'cnv_levels': 3,
                 'mutation_binary': True,
                 'balanced_data': False,
                 'combine_type': 'union',
                 'use_coding_genes_only': True,
                 'selected_genes': selected_genes,
                 'training_split': 0
                 }
             }

data = []

splits = np.arange(0, 1)
for n in splits:
    d = copy.deepcopy(data_base)
    d['id'] = 'data_{}'.format(n)
    d['params']['training_split'] = str(n)
    data.append(d)

pre = {'type': None}

features = {}

class_weight = {0: 0.75, 1: 1.5}
models = [
    {
        'type': 'sgd',
        'id': 'L2 Logistic Regression',
        'params': {'loss': 'log', 'penalty': 'l2', 'alpha': 0.01, 'class_weight': class_weight}
    },

    {
        'type': 'svc',
        'id': 'RBF Support Vector Machine ',
        'params': {'kernel': 'rbf', 'C': 100, 'gamma': 0.001, 'probability': True, 'class_weight': class_weight}
    },

    {
        'type': 'svc', 'id':
        'Linear Support Vector Machine ',
        'params': {'kernel': 'linear', 'C': 0.1, 'probability': True, 'class_weight': class_weight}
    },

    {
        'type': 'random_forest',
        'id': 'Random Forest',
        'params': {'max_depth': None, 'n_estimators': 50, 'bootstrap': False, 'class_weight': class_weight}
    },

    {
        'type': 'adaboost',
        'id': 'Adaptive Boosting',
        'params': {'learning_rate': 0.1, 'n_estimators': 50}
    },

    {
        'type': 'decision_tree',
        'id': 'Decision Tree',
        'params': {'min_samples_split': 10, 'max_depth': 10}
    },

]

pipeline = {'type': 'crossvalidation', 'params': {'n_splits': 5, 'save_train': True}}