# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 14:38:38 2021

@author: Dell
"""

import os
import re
import pandas as pd


class GMTReader:
    """
    Reads a Gene Matrix Transposed File (.gmt) that contains a neural network specifications.
    
    From the paper:
        Note that the P-NET model is not bound to a certain
        architecture, as the model architecture is automatically built by reading
        model specifications provided by the user via a gene matrix transposed
        file format (.gmt) file, and custom pathways, gene sets and modules
        with custom hierarchies can be provided by the user.
    """
    
    def __init__(self):
        return
    
    def load_data(self, filename, genes_col=0, pathway_col=0):

        data_dict_list = []
        with open(filename) as gmt:

            data_list = gmt.readlines()

            # print data_list[0]
            for row in data_list:
                genes = row.strip().split('\t')
                genes = [re.sub('_copy.*', '', g) for g in genes]
                genes = [re.sub('\\n.*', '', g) for g in genes]
                for gene in genes[genes_col:]:
                    pathway = genes[pathway_col]
                    dict = {'group': pathway, 'gene': gene}
                    data_dict_list.append(dict)

        df = pd.DataFrame(data_dict_list)

        return df