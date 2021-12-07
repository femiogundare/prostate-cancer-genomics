# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 11:02:16 2021

@author: femiogundare
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from data_scripts.prostate.data_reader import ProstateData


class Data:
    """
    Data class.
    """
    def __init__(self, id, type, params, test_size=0.3, stratify=True):
        """
        Parameters:
            id: str
                The dataset ID.
                
            type: str, default prostate_data
                The type of dataset. The only possible value is prostate_data.
                
            params: dict
                A dictionary that contains parameters of the ProstateData class from data_reader script. These parameters include 
                data_type, drop_AR, mutation_binary, balanced_data, combine_type, etc.
        """
        self.data_type = type
        self.data_params = params
        self.test_size = test_size
        self.stratify = stratify
        
        if self.data_type == 'prostate_data':
            self.data_reader = ProstateData(**self.data_params)
        else:
            logging.error('Unsupported data type')
            raise ValueError('unsupported data type')
    
    def get_train_val_test(self):
        return self.data_reader.get_train_val_test()
    
    def get_train_test(self):
        X_train, X_val, X_test, y_train, y_val, y_test, train_samples, val_samples, test_samples, genes = self.data_reader.get_train_val_test()
        # Combine training and validation sets
        X_train = np.concatenate((X_train, X_val))
        y_train = np.concatenate((y_train, y_val))
        train_samples = list(train_samples) + list(val_samples)
        return X_train, X_test, y_train, y_test, train_samples, test_samples, genes
    
    def get_data(self):
        x = self.data_reader.x
        samples = self.data_reader.samples
        response = self.data_reader.response
        genes = self.data_reader.genes
        return x, samples, response, genes
    
    def get_relevant_features(self):
        if hasattr(self.data_reader, 'relevant_features'):
            return self.data_reader.get_relevant_features()
        else:
            return None
        
        
        
if __name__ == '__main__':
    selected_genes = 'tcga_prostate_expressed_genes_and_cancer_genes_and_memebr_of_reactome.csv'
    selected_samples = 'samples_with_fusion_data.csv'
    data_params = {'id': 'ALL', 'type': 'prostate_data',
           'params': {
               'data_type': ['important_mutations', 'cnv_deletion', 'cnv_amplification'],
               'account_for_data_type': ['TMB'],
               'drop_AR': False,
               'cnv_levels': 3,
               'mutation_binary': False,
               'balanced_data': False,
               'combine_type': 'union',  # intersection
               'use_coding_genes_only': True,
               'selected_genes': selected_genes,
               'selected_samples': None,
               'training_split': 0,
               }
           }
    
    data = Data(**data_params)
    x, samples, response, genes = data.get_data()
    print(x.shape[0], len(samples), response.shape[0], len(genes))
    
    X_train, X_test, y_train, y_test, train_samples, test_samples, genes = data.get_train_test()
    
    X_train_df = pd.DataFrame(X_train, columns=genes, index=train_samples)
    
    print(genes.levels)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    print(X_train.sum().sum())
    
    x, samples, response, genes = data.get_data()
    x_df = pd.DataFrame(x, columns=genes, index=samples)
    print(x_df.shape)
    print(x_df.sum().sum())