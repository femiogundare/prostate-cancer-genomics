# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 10:37:01 2021

@author: femiogundare
"""

import os
import numpy as np
import pandas as pd
import logging
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from data_scripts.gmt_reader import GMTReader
from config import PATHWAY_PATH


def get_KEGG_map(input_list, filename, genes_col=1, shuffle_genes=False):
    """
    Args:
        input_list: a list of inputs under consideration (e.g genes)
        filename: a gmt formatted file e.g. pathway1 gene1 gene2 gene3
                                            pathway2 gene4 gene5 gene6
        genes_col: the start index of the genes column
        shuffle_genes: [True or False]
    
    Returns:
        A tuple of map values 1 or 0 based on the membership of certain gene in the corresponding pathway, genes, and pathways.
        
    '''
    """
    gmt_reader = GMTReader()
    df = gmt_reader.load_data(filename, genes_col)
    df['value'] = 1
    mapp = pd.pivot_table(df, values='value', index='gene', columns='group', aggfunc=np.sum)
    mapp = mapp.fillna(0)
    cols_df = pd.DataFrame(index=input_list)
    mapp = cols_df.merge(mapp, right_index=True, left_index=True, how='left')
    mapp = mapp.fillna(0)
    genes = mapp.index
    pathways = mapp.columns
    mapp = mapp.values
    
    if shuffle_genes:
        logging.info('shuffling')
        ones_ratio = np.sum(mapp) / np.prod(mapp.shape)
        logging.info('ones_ratio {}'.format(ones_ratio))
        mapp = np.random.choice([0, 1], size=mapp.shape, p=[1 - ones_ratio, ones_ratio])
        logging.info('random map ones_ratio {}'.format(ones_ratio))
        
    return mapp, genes, pathways



if __name__ == '__main__':
    input_genes = ['AR', 'AKT', 'EGFR']
    filename = os.path.join(PATHWAY_PATH, 'MsigDB', 'c2.cp.kegg.v6.1.symbols.gmt')
    mapp, genes, pathways = get_KEGG_map(input_list=input_genes, filename=filename)
    print(genes)
    print(pathways)
    print(mapp)