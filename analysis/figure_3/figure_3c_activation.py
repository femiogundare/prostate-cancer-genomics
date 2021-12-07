# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 00:32:51 2021

@author: femiogundare
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from data_extraction_utils import get_pathway_names

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import PATHWAY_PATH, PLOTS_PATH

extracted_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), '_extracted')
saving_dir = os.path.join(PLOTS_PATH, 'figure3/activations')
if not os.path.exists(saving_dir):
    os.makedirs(saving_dir)
    

def plot_activation(ax, l, column='coef_combined', layer=3, pad=200):
    node_activation = pd.read_csv(os.path.join(extracted_dir, 'node_importance_graph_adjusted.csv'), index_col=0)
    response = pd.read_csv(os.path.join(extracted_dir, 'response.csv'), index_col=0)
    df = pd.read_csv(os.path.join(extracted_dir, 'activation_{}.csv'.format(layer)), index_col=0)
    df.columns = get_pathway_names(df.columns)
    
    if layer == 1:
        column = 'coef_combined'
        
    high_nodes = node_activation[node_activation.layer == layer].abs().nlargest(10, columns=[column])
    high_nodes = high_nodes.sort_values(column, ascending=False)
    features = list(high_nodes.index)
    to_be_saved = df[features].copy()

    y = response.reindex(to_be_saved.index)
    df = to_be_saved.copy()
    features = list(to_be_saved.columns)
    df["group"] = y
    df2 = pd.melt(df, id_vars='group', value_vars=features, value_name='value')
    df2['group'] = df2['group'].replace(0, 'Primary')
    df2['group'] = df2['group'].replace(1, 'Metastatic')

    def short_names(name):
        if len(name) > 55:
            ret = name[:55] + '...'
        else:
            ret = name
        return ret

    df2.variable = df2['variable'].apply(short_names)
    sns.set_color_codes('muted')
    current_palette = sns.color_palette()
    palette = dict(Primary=current_palette[0], Metastatic=current_palette[1])
    sns.violinplot(y="variable", x="value", hue="group", data=df2, split=True, bw=.3, inner=None, palette=palette,
                   linewidth=.5, ax=ax)
    ax.autoscale(enable=True, axis='x', tight=True)

    ax.get_legend().remove()
    ax.set_xlim((-1.2, 1.2))
    fontProperties = {'family': 'Arial', 'weight': 'normal', 'size': 10}
    ax.set_yticklabels(ax.get_yticklabels(), fontProperties)

    ax.set_ylabel('')
    ax.set_xlabel('Activation', fontdict=dict(family='Arial', weight='bold', fontsize=14))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    for tick in ax.yaxis.get_major_ticks():
        tick.tick1line.set_markersize(0)
        tick.tick2line.set_markersize(0)
        tick.label1.set_horizontalalignment('left')

    ax.tick_params(axis="y", direction="out", pad=pad)
    ax.yaxis.set_ticks_position('none')

    ax.set_ylabel('Layer {}'.format(l), fontdict=dict(family='Arial', weight='bold', fontsize=14))
    
    
    
def run():
    left_adjust = [0.1, 0.]
    pad = [50, 250, 200, 250, 200, 200]
    for l in range(1, 7):
        fig = plt.figure(figsize=(9, 4))
        ax = fig.subplots(1, 1)
        plot_activation(ax, l, column='coef', layer=l, pad=pad[l - 1])
        if l == 1:
            shift = 0.3
        else:
            shift = 0.6
            
        plt.subplots_adjust(left=shift)
        filename = os.path.join(saving_dir, 'layer_{}_activation.png'.format(l))
        plt.savefig(filename, dpi=200)
        plt.close()
        
        
if __name__ == '__main__':
    run()