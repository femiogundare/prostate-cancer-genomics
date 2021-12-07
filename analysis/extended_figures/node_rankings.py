# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 04:42:23 2021

@author: femiogundare
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from config import PROSTATE_LOG_PATH, PLOTS_PATH

extracted_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), '_extracted')
saving_dir = os.path.join(PLOTS_PATH, 'extended_figures')
if not os.path.exists(saving_dir):
    os.makedirs(saving_dir)


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
        ax.set_xticklabels(ax.get_xticklabels(), fontproperties)
        ax.set_xlabel('')
        # ax.set_ylabel('')
        ax.set_ylabel('Importance Score', fontdict=fontproperties)
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
        
    elif graph == 'barplot_sum':
        def shorten_name(name):
            if len(name) >= 40:
                name = name[:40] + ' ...'
            return name

        df2['variable'] = df2['variable'].apply(shorten_name)
        df2 = df2[df2.value != 0]

        def abs_sum_estimator(ins):
            return np.abs(np.sum(ins))

        ax = sns.barplot(y="variable", x="value", data=df2, estimator=abs_sum_estimator, n_boot=1000, ci=95, ax=ax,
                         color="b", errwidth=1, palette="Blues_d")
        sns.despine(left=True, bottom=True)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=fontsize, horizontalalignment="left")
        ax.tick_params(axis='both', which='major', pad=150)
        if layer == 6:
            ax.set_xlabel('Total importance score', fontsize=fontsize)
        else:
            ax.set_xlabel('')

        ax.set_ylabel('Layer H{}'.format(layer), fontproperties, labelpad=20)
        ax.set_xticks([], [])
        ax.set_yticks(ax.get_yticks(), [])
        ax.tick_params(axis=u'both', which=u'both', length=0)
        
        
def shorten_names(name):
    if len(name) >= 60:
        name = name[:60] + '...'
    return name


fontsize = 5  # legends, axis
fontproperties = {'family': 'Arial', 'weight': 'normal', 'size': 6}

def plot_axis(axis):
    node_importance = pd.read_csv(os.path.join(extracted_dir, 'node_importance_graph_adjusted.csv'),
                                  index_col=0)
    response = pd.read_csv(os.path.join(extracted_dir, 'response.csv'), index_col=0)
    layers = sorted(list(node_importance.layer.unique()))

    for ax, l in zip(axis, layers):
        if l == 1:
            plot_high_genes2(ax, layer=l, graph='barplot_sum')
        else:
            plot_high_genes2(ax, layer=l, graph='barplot_sum')
            
            
def run():
    fig = plt.figure(constrained_layout=False, figsize=(7.2, 9.72))
    spec2 = gridspec.GridSpec(ncols=1, nrows=6, figure=fig)
    ax1 = fig.add_subplot(spec2[0, 0])
    ax2 = fig.add_subplot(spec2[1, 0])
    ax3 = fig.add_subplot(spec2[2, 0])
    ax4 = fig.add_subplot(spec2[3, 0])
    ax5 = fig.add_subplot(spec2[4, 0])
    ax6 = fig.add_subplot(spec2[5, 0])

    plot_axis([ax1, ax2, ax3, ax4, ax5, ax6])
    fig.tight_layout()

    plt.gcf().subplots_adjust(left=0.5, right=0.8, bottom=0.15)
    filename = os.path.join(saving_dir, 'node_rankings.png')
    plt.savefig(filename, dpi=300)


if __name__ == "__main__":
    run()