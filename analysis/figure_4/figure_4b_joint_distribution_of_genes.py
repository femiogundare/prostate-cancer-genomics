# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 22:35:23 2021

@author: femiogundare
"""

"""
This script contains the code to display the joint distribution of genes AR, TP53 and MDM4 alterations across 1013 prostate 
cancer samples using an UpSetPlot.

A gene is said to be altered if it has a mutation, deep deletion or high amplification.

In the paper, Integrative clinical genomics of metastatic cancer, Robinson et al. found the TP53 to be the most frequently altered 
tumor suppressor gene and AR to be the second most frequently mutated oncogene.
"""


import os
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from upsetplot import UpSet
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from config import PROSTATE_DATA_PATH, PLOTS_PATH
from data_scripts.data_access import Data



selected_genes = ['AR', 'TP53', 'MDM4', 'CDK4', 'CDK6', 'CDKN2A', 'RB1']
data_params = {'data_type': ['important_mutations', 'cnv_deletion', 'cnv_amplification'],
               'drop_AR': False,
               'cnv_levels': 5,
               'mutation_binary': True,
               'balanced_data': False,
               'combine_type': 'union',  # intersection
               'use_coding_genes_only': False,
               'selected_genes': selected_genes}

data_access_params = {'id': 'id', 'type': 'prostate_data', 'params': data_params}
data_adapter = Data(**data_access_params)
x, samples, response, genes = data_adapter.get_data()
x_df = pd.DataFrame(x, columns=genes, index=samples)
x_df.head()

x_df_3 = x_df.copy()
x_df_3[x_df_3 < 1] = 0  # Remove single copy

x_df_3 = x_df_3.T.reset_index().groupby('level_0').sum()  # all events (OR)
x_df_3[x_df_3 > 0] = 1  # Binarize


x_df_3_binary = x_df_3.T > 0.
x_df_3_binary = x_df_3_binary.set_index(selected_genes)

y_ind = response > 0
x_df_mets_3 = x_df_3.T[y_ind].T

x_df_mets_3_binary = x_df_mets_3.T > 0.

x_df_mets_3_binary = x_df_mets_3_binary.set_index(selected_genes)

font = {'family': 'Arial',
        'weight': 'normal',
        'size': 5}
matplotlib.rc('font', **font)
dd = x_df_3_binary.reset_index().set_index(['AR', 'TP53', 'MDM4'])

upset = UpSet(dd, subset_size='count', intersection_plot_elements=6, show_counts=True, with_lines=True, element_size=10)
fig = plt.figure(constrained_layout=False, figsize=(8, 6))
upset.plot(fig)
fig.subplots_adjust(bottom=0.2, top=0.9, left=0.08, right=0.99)

saving_dir = os.path.join(PLOTS_PATH, 'figure4')
filename = os.path.join(saving_dir, 'joint_distribution_of_AR_TP53_MDM4.png')
plt.savefig(filename, dpi=300)