# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 21:36:40 2021

@author: femiogundare
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from config import PLOTS_PATH
from data_scripts.data_access import Data


data_params = {'id': 'cnv', 'type': 'prostate_data', 'params': {'data_type': 'cnv', 'drop_AR': False}}
data = Data(**data_params)
x, samples, response, genes = data.get_data()
df_cnv = pd.DataFrame(x, index=samples, columns=list(genes))
df_cnv['y'] = response

data_params = {'id': 'mut', 'type': 'prostate_data', 'params': {'data_type': 'important_mutations', 'drop_AR': False}}
data = Data(**data_params)
x, samples, response, genes = data.get_data()
df_mut = pd.DataFrame(x, index=samples, columns=list(genes))
df_mut['y'] = response


def plot_stacked_hist_cnv(df, gene_name, ax):
    ind = np.sort(df[gene_name].unique())
    primary_df = df[df['y'] == 0]
    mets_df = df[df['y'] == 1]
    p = primary_df[gene_name].value_counts()
    m = mets_df[gene_name].value_counts()

    mapping = {0: 'Neutral', 1: 'Amplification', 2: 'High amplification', -1: 'Deletion', -2: 'Deep deletion'}

    index = pd.DataFrame(index=ind)
    summary = pd.concat([m, p], axis=1, keys=['Metastatic', 'Primary'])
    summary = summary.join(index, how='right')
    summary.fillna(0, inplace=True)
    summary = 100. * summary / summary.sum()

    summary = summary.rename(index=mapping)
    D_id_color = {'High amplification': 'maroon', 'Amplification': 'lightcoral', 'Neutral': 'gainsboro',
                  'Deletion': 'skyblue', 'Deep deletion': 'steelblue'}

    color = [D_id_color[i] for i in summary.index]
    bars = summary.T.plot.bar(stacked=True, ax=ax, color=color)

    ax.tick_params(axis='x', rotation=0, labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(fontsize=8, bbox_to_anchor=(.8, -0.1))

    ax.set_ylabel('Sample percent (%)', fontdict=dict(family='Arial', weight='bold', fontsize=12))

    ax.set_title('Copy number variations', fontdict=dict(family='Arial', weight='normal', fontsize=10))
    
    
    
def plot_stacked_hist_mut(df, gene_name, ax):
    ind = np.sort(df[gene_name].unique())
    primary_df = df[df['y'] == 0]
    mets_df = df[df['y'] == 1]
    p = primary_df[gene_name].value_counts()
    m = mets_df[gene_name].value_counts()

    index = pd.DataFrame(index=ind)
    summary = pd.concat([m, p], axis=1, keys=['Metastatic', 'Primary'])
    summary = summary.join(index, how='right')
    summary.fillna(0, inplace=True)
    summary = 100. * summary / summary.sum()

    D_id_color = {0: 'gainsboro', 3: 'maroon', 1: 'lightcoral', 2: 'red'}

    current_palette = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=False)

    colors = [D_id_color[int(c)] for c in summary.index]
    bars = summary.T.plot.bar(stacked=True, ax=ax, color=colors)

    ax.tick_params(axis='x', labelrotation=0.)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(fontsize=8, bbox_to_anchor=(.7, -0.1))

    ax.set_ylabel('Sample percent (%)', fontdict=dict(family='Arial', weight='bold', fontsize=12))

    ax.set_title('Number of mutations', fontdict=dict(family='Arial', weight='normal', fontsize=10))
    

selected_genes = ['AR', 'TP53', 'PTEN', 'FGFR1', 'MDM4', 'RB1', 'NOTCH1', 'MAML3', 'PDGFA', 'EIF3E']  
saving_dir = os.path.join(PLOTS_PATH, 'figure4/sample_percent')
if not os.path.exists(saving_dir):
    os.makedirs(saving_dir)  
    
def run():
    for gene in selected_genes:
        fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(6, 4), dpi=200)
        plt.ylim(0, 110)
        plt.subplots_adjust(bottom=0.3)

        if gene in df_cnv.columns:
            plot_stacked_hist_cnv(df_cnv, gene, axes[0])
        if gene in df_mut.columns:
            plot_stacked_hist_mut(df_mut, gene, axes[1])

        filename = os.path.join(saving_dir, gene + '.png')
        fig.suptitle(gene, fontdict=dict(family='Arial', weight='bold', fontsize=12))
        plt.savefig(filename, dpi=200)
        plt.close()
        
        
if __name__ == "__main__":
    run()