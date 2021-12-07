# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 12:17:11 2021

@author: femiogundare
"""

import os
import pandas as pd
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import REACTOME_PATHWAY_PATH


def get_reactome_pathway_names():
    """
    Returns a dataframe containing reactome ids, names of reactome pathways, and specie (homo sapiens).
    """
    reactome_pathways_df = pd.read_csv(os.path.join(REACTOME_PATHWAY_PATH, 'ReactomePathways.txt'), sep='	', header=None)
    reactome_pathways_df.columns = ['reactome_id', 'pathway_name', 'specie']
    reactome_pathways_df_human = reactome_pathways_df[reactome_pathways_df['specie'] == 'Homo sapiens']
    reactome_pathways_df_human.reset_index(inplace=True)
    return reactome_pathways_df_human