# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 23:07:06 2021

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

extracted_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), '_extracted')
saving_dir = os.path.join(PLOTS_PATH, 'figure3/gene_importance')
if not os.path.exists(saving_dir):
    os.makedirs(saving_dir)
    
    
def shorten_names(name):
    if len(name) >= 60:
        name = name[:60] + '...'
    return name
    

def plot_high_genes(df, name, col='avg_score'):
    df.index = df.index.map(shorten_names)
    x_pos = range(df.shape[0])
    ax = sns.barplot(y=df.index, x=col, data=df,
                     palette="Blues_d")
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.set_title('Layer{}'.format(name))
    ax.set_xticks([], [])
    ax.set_yticks(ax.get_yticks(), [])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    
def plot_jitter(group_col, val_col, data, ax):
    vals = data.groupby(group_col)[val_col]
    sums = vals.sum().to_frame().sort_values(val_col, ascending=True)
    inds = np.arange(1, len(sums) + 1)
    for i, s in zip(inds, sums.index):
        ind = data[group_col] == s
        n = sum(ind)
        x = data.loc[ind, val_col]
        y = np.array([i - 0.3] * n)
        noise = np.random.normal(0, 0.02, n)
        y = y + noise
        ax.plot(x, y, '.', markersize=5)
    
    
def boxplot_csutom(group_col, val_col, data, ax):
    vals = data.groupby(group_col)[val_col]
    # quartile1 = vals.quantile(0.25)
    # medians = vals.quantile(0.5)
    # quartile3 = vals.quantile(0.75)
    #
    # mins = vals.min()
    # maxs = vals.max()

    sums = vals.sum().to_frame().sort_values(val_col, ascending=True)
    quartile1 = vals.quantile(0.25).reindex(sums.index)
    medians = vals.quantile(0.5).reindex(sums.index)
    quartile3 = vals.quantile(0.75).reindex(sums.index)

    mins = vals.min().reindex(sums.index)
    maxs = vals.max().reindex(sums.index)

    def adjacent_values(mins, maxs, q1, q3):
        upper_adjacent_value = q3 + (q3 - q1) * 1.5
        upper_adjacent_value = np.clip(upper_adjacent_value, q3, maxs)

        lower_adjacent_value = q1 - (q3 - q1) * 1.5
        lower_adjacent_value = np.clip(lower_adjacent_value, mins, q1)
        return lower_adjacent_value, upper_adjacent_value

    whiskers = np.array([adjacent_values(mi, mx, q1, q3) for mi, mx, q1, q3 in zip(mins, maxs, quartile1, quartile3)])

    inds = np.arange(1, len(medians) + 1)
    whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]
    ax.hlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
    ax.hlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

    # xticks = ax.xaxis.get_major_ticks()
    # xticks = ax.xaxis.get_minor_ticks()
    # xticks[0].label1.set_visible(False)
    # xticks[-1].label1.set_visible(False)

    # ax.set_yticks(inds)
    # ax.set_yticklabels(medians.index)

    
    
def plot_high_genes2(ax, layer=1, graph='hist', direction='h'):
    if layer == 1:
        column = 'coef_combined'
    else:
        column = 'coef'

    node_importance = pd.read_csv(os.path.join(extracted_dir, 'node_importance_graph_adjusted.csv'), index_col=0)
    high_nodes = node_importance[node_importance.layer == layer].abs().nlargest(10, columns=[column])
    features = list(high_nodes.index)
    
    response = pd.read_csv(os.path.join(extracted_dir, 'response.csv'), index_col=0)
    df_in = pd.read_csv(os.path.join(extracted_dir, 'gradient_importance_detailed_{}.csv').format(layer), index_col=0)
    df_in = df_in.copy()
    df_in = df_in.join(response)
    df_in['group'] = df_in.response
    df2 = pd.melt(df_in, id_vars='group', value_vars=list(features), value_name='value')

    if graph == 'hist':
        df2 = pd.melt(df_in, id_vars='group', value_vars=list(features), value_name='value')
        bins = np.linspace(df2.value.min(), df2.value.max(), 20)
        g = sns.FacetGrid(df2, col="variable", hue="group", col_wrap=2)
        g.map(plt.hist, 'value', bins=bins, ec="k")
        g.axes[-1].legend(['Primary', 'Metastatic'])
        
    elif graph == 'viola':
        sns.violinplot(x="variable", y="value", hue="group", data=df2, split=True, bw=.6, inner=None, ax=ax)
        ax.legend(['Primary', 'Metastatic'])
        fontProperties = dict(family='Arial', weight='normal', size=14, rotation=30, ha='right')
        ax.set_xticklabels(ax.get_xticklabels(), fontProperties)
        ax.set_xlabel('')
        # ax.set_ylabel('')
        ax.set_ylabel('Importance Score', fontdict=dict(family='Arial', weight='bold', fontsize=14))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

    elif graph == 'swarm':
        df2 = pd.melt(df_in, id_vars='group', value_vars=list(features), value_name='value')
        df2['group'] = df2['group'].replace(0, 'Primary')
        df2['group'] = df2['group'].replace(1, 'Metastatic')
        df2.value = df2.value.abs()

        current_palette = sns.color_palette()
        ax = sns.swarmplot(x="variable", y="value", data=df2, hue="group",
                           palette=dict(Primary=current_palette[0], Metastatic=current_palette[1]), ax=ax)
        plt.setp(ax.get_legend().get_texts(), fontsize='14')  # for legend text
        fontProperties = dict(family='Arial', weight='normal', size=12, rotation=30, ha='right')
        ax.set_xticklabels(ax.get_xticklabels(), fontProperties)
        # ax.tick_params(labelsize=10)
        ax.set_xlabel('')
        ax.set_ylabel('Importance Score', fontdict=dict(family='Arial', weight='bold', fontsize=14))
        ax.legend().set_title('')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

    elif graph == 'boxplot_custom':
        fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 3]})

        df2 = pd.melt(df_in, id_vars='group', value_vars=list(features), value_name='value')
        df2['group'] = df2['group'].replace(0, 'Primary')
        df2['group'] = df2['group'].replace(1, 'Metastatic')
        df2.value = df2.value.abs()

        sums = df2.groupby('variable')['value'].sum().sort_values(ascending=False).to_frame()

        ax1 = sns.barplot(y='variable', x='value', data=sums.reset_index(), palette="Blues_d", ax=ax1)
        ax1.invert_xaxis()
        ax1.set_xscale('log')
        ax1.set_ylabel('')
        ax1.set_xlabel('')
        ax1.set_xticks([], [])
        ax1.set_yticks(ax1.get_yticks(), [])
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        # ax1.spines['bottom'].set_visible(False)
        ax1.set_xlabel('Total importance score', labelpad=15, fontdict=dict(family='Arial', weight='bold', fontsize=12))
        ax1.spines['left'].set_visible(False)
        # ax1.tick_params(bottom='off', which='both')
        ax1.tick_params(left='off', which='both')

        df2 = df2[df2.value != 0]
        boxplot_csutom(val_col="value", group_col="variable", data=df2, ax=ax2)
        plot_jitter(val_col="value", group_col="variable", data=df2, ax=ax2)

        ax2.set_ylabel('')
        ax2.set_xlabel('Sample-level importance score', fontdict=dict(family='Arial', weight='bold', fontsize=12))

        # ax2.set_xticks([], [])
        ax2.set_xlim(0, 1)
        ax2.set_yticks([], [])
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        # ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        
        
        
def run():
    node_importance = pd.read_csv(os.path.join(extracted_dir, 'node_importance_graph_adjusted.csv'), index_col=0)
    response = pd.read_csv(os.path.join(extracted_dir, 'response.csv'), index_col=0)
    layers = list(node_importance.layer.unique())
    
    plt.close()
    fig = plt.figure(figsize=(8, 4), dpi=200)
    ax = fig.subplots(1, 1)
    plot_high_genes2(ax, layer=1, graph='boxplot_custom')
    filename = os.path.join(saving_dir, 'high_genes.png')
    plt.savefig(filename, dpi=200)
    plt.close()
    
    for layer in layers:
        fig = plt.figure(figsize=(8, 4), dpi=200)
        if layer == 1:
            high_nodes = node_importance[node_importance.layer == layer].abs().nlargest(10, columns=['coef_combined'])
            plot_high_genes(high_nodes, name=str(layer), col='coef_combined')
        else:
            high_nodes = node_importance[node_importance.layer == layer].abs().nlargest(10, columns=['coef'])
            plot_high_genes(high_nodes, name=str(layer), col='coef')

        if layer == 2:
            shift = 0.7
        else:
            shift = 0.6

        plt.gcf().subplots_adjust(left=shift)
        filename = os.path.join(saving_dir, 'layer_' + str(layer) + '_high_genes.png')
        plt.savefig(filename)
        high_nodes.to_csv(os.path.join(saving_dir, 'layer_{}_high_genes.csv'.format(layer)))
        
        
if __name__ == '__main__':
    run()