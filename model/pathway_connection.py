# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 07:16:26 2021

@author: femiogundare
"""

import os
import numpy as np
import pandas as pd
import scipy

from data_scripts.data_access import Data
from data_scripts.gmt_reader import GMTReader


def get_map(data, gene_dict, pathway_dict):
    genes = data['gene']
    pathways = data['group'].fillna('')
    
    n_genes = len(gene_dict)
    n_pathways = len(pathway_dict) + 1
    n = data.shape[0]
    row_index = np.zeros((n, ))
    col_index = np.zeros((n, ))
    
    for i, gene in enumerate(genes):
        row_index[i] = gene_dict[gene]
        
    for i, pathway in enumerate(pathways):
        if pathway == '':
            col_index[i] = n_pathways - 1
        else:
            col_index[i] = pathway_dict[pathway]
            
    #print('Number of genes: {}'.format(n_genes))
    #print('Number of pathways: {}'.format(n_pathways))
    #print(np.max(col_index))
    
    mapp = scipy.coo_matrix(([1] * n, (row_index, col_index)), shape=(n_genes, n_pathways))
    return mapp


def get_dict(array_of_values):
    unique_list = np.unique(array_of_values)
    output_dict = {}
    
    for i, gene in unique_list:
        output_dict[gene] = i
    return output_dict

"""
def get_connection_map(data_params):
    data = Data(**data_params)
    x, samples, response, genes = data.get_data()
    x = pd.DataFrame(x.T, index=genes)
    
    gmt_reader = GMTReader()
"""