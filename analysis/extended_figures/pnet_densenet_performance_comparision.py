# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 00:39:35 2021

@author: femiogundare
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt, gridspec
from matplotlib.ticker import FormatStrFormatter, NullFormatter
from scipy import stats
from sklearn import metrics
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from config import PROSTATE_LOG_PATH, PROSTATE_DATA_PATH, PLOTS_PATH


def get_densenet_sameweights(col='f1'):
    filename = os.path.join(PROSTATE_LOG_PATH, 'number_samples/crossvalidation_number_samples_densenet_sameweights')
    filename = filename + '/folds.csv'
    df = pd.read_csv(filename, index_col=0, header=[0, 1])
    
    dd = df.swaplevel(0, 1, axis=1)[col].head()
    df_densenet_col = [c for c in dd.columns if 'dense' in c]
    df_densenet = dd[df_densenet_col]
    
    return df_densenet


def get_pnet(col='f1'):
    filename = os.path.join(PROSTATE_LOG_PATH, 'number_samples/crossvalidation_average_reg_10_tanh')
    filename = filename + '/folds.csv'
    df = pd.read_csv(filename, index_col=0, header=[0, 1])
    dd = df.swaplevel(0, 1, axis=1)[col].head()
    df_pnet_col = [c for c in dd.columns if 'P-net' in c]
    df_pnet = dd[df_pnet_col]
    
    return df_pnet


def plot_comparison(ax1, label, df_pnet, df_dense):
    y1 = df_pnet.mean()
    dy = df_pnet.std()
    x = sizes
    sns.set_color_codes('muted')
    current_palette = sns.color_palette()
    colors = current_palette[0:2]
    ax1.plot(x, y1, linestyle='-', marker='o', color=colors[0])
    ax1.fill_between(x, y1 - dy, y1 + dy, color=colors[0], alpha=0.2)
    y2 = df_dense.mean()
    dy = df_dense.std()
    ax1.plot(x, y2, linestyle='-', marker='o', color=colors[1])
    ax1.fill_between(x, y2 - dy, y2 + dy, color=colors[1], alpha=0.1)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.set_ylabel(label, fontdict=fontproperties)
    ax1.legend(['P-NET', 'Dense'], fontsize=8, loc='upper left', framealpha=0)
    
    ax1.set_xticklabels(ax1.get_xticklabels(), fontsize=fontsize)
    ax1.set_yticklabels(ax1.get_yticks(), size=fontsize)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax1.set_ylim(0.3, 1.0)
    
    
sizes = []
for i in range(0, 20, 3):
    df_split = pd.read_csv(os.path.join(PROSTATE_DATA_PATH, 'splits/training_set_{}.csv'.format(i)), index_col=0)
    sizes.append(df_split.shape[0])
    
    
def get_stats(df_pnet, df_densenet):
    p_values = []
    
    for col_1, col_2 in zip(df_pnet.columns, df_densenet.columns):
        x = df_pnet.loc[:, col_1]
        y = df_densenet.loc[:, col_2]
        
        twosample_results = stats.ttest_ind(x, y)
        p_value = twosample_results[1] / 2
        p_values.append(p_value)
        
    return p_values


cols = ['f1', 'aupr', 'precision', 'recall', 'accuracy']

labels = ['F1 score', 'Area Under Precision Recall Curve', 'Precision', 'Recall',
          'Accuracy']


def plot_pnet_vs_densenet(ax, c, label, plot_ratio=False):
    sns.set_color_codes('muted')
    current_palette = sns.color_palette()
    color = current_palette[3]

    sizes = []
    for i in range(0, 20, 3):
        df_split = pd.read_csv(os.path.join(PROSTATE_DATA_PATH, 'splits/training_set_{}.csv'.format(i)), index_col=0)
        sizes.append(df_split.shape[0])
    sizes = np.array(sizes)
    
    df_densenet_sameweights = get_densenet_sameweights(c)
    df_pnet = get_pnet(col=c)
    p_values = get_stats(df_pnet, df_densenet_sameweights)
    plot_comparison(ax, label, df_pnet, df_densenet_sameweights)
    
    updated_values = []
    for i, (p, s) in enumerate(zip(p_values, sizes)):
        if p >= 0.05:
            displaystring = r'n.s.'
        elif p < 0.0001:
            displaystring = r'***'
        elif p < 0.001:
            displaystring = r'**'
        else:
            displaystring = r'*'
        updated_values.append('{:.0f}\n({})'.format(s, displaystring))
        ax.axvline(x=s, ymin=0, linestyle='--', alpha=0.3)
        
    ax.set_xscale("log")
    ax.set_xticks([], [])
    
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis=u'x', which=u'both', length=0)
    
    ax.set_xticks(sizes)
    ax.set_xticklabels(updated_values)
    ax.set_xlim((min(sizes) - 5, max(sizes) + 50))
    
    if plot_ratio:
        ax2 = ax.twinx()
        y1 = df_pnet.mean()
        y2 = df_densenet_sameweights.mean()
        ratio = (y1.values - y2.values) / y2.values
        new_x = np.linspace(min(sizes), max(sizes), num=np.size(sizes))
        coefs = np.polyfit(sizes, ratio, 3)
        new_line = np.polyval(coefs, new_x)
        
        ax2.plot(new_x, new_line, '-.', linewidth=0.5, color=color)
        ax2.set_ylim((0.005, .23))
        ax.set_ylim((.5, 1.05))
        ax2.set_ylabel('Performance increase', fontproperties)
        vals = ax2.get_yticks()
        ax2.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
        ax2.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
        ax.set_yticks([], minor=True)
        ax2.spines['right'].set_color(color)
        ax2.yaxis.label.set_color(color)
        ax2.tick_params(axis='y', colors=color)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)

    ax.set_xlabel('Number of samples', fontproperties)
    size_vals = ax.get_xticks()
    pvalues_dict = {}
    for p, s in zip(p_values, sizes):
        pvalues_dict[s] = p
        
    return pvalues_dict



def plot_pnet_vs_dense_same_arch(ax):
    def plot_roc(ax, y_test, y_pred_score, save_dir, color, label=''):
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_score, pos_label=1)
        roc_auc = metrics.auc(fpr, tpr)

        ax.plot(fpr, tpr, label=label + ' (area = %0.2f)' % roc_auc, linewidth=1, color=color)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.1, linewidth=1.)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])

        ax.set_xlabel('False Positive Rate', fontproperties)
        ax.set_ylabel('True Positive Rate', fontproperties)

    pnet_base_dir = os.path.join(PROSTATE_LOG_PATH, 'prostate_net/onesplit_average_reg_10_tanh_test')
    df_pnet = pd.read_csv(os.path.join(pnet_base_dir, 'P-net_ALL_testing.csv'), sep=',', index_col=0, header=[0])

    models_base_dir = os.path.join(PROSTATE_LOG_PATH, 'dense_net/onesplit_dense_test')
    #df_dense = pd.read_csv(os.path.join(models_base_dir, 'P-net_ALL_testing.csv'), sep=',', index_col=0, header=[0])
    df_dense = pd.read_csv(os.path.join(models_base_dir, 'dense_data_0_testing.csv'), sep=',', index_col=0, header=[0])

    y_test = df_pnet['y']
    y_pred_score = df_pnet['pred_scores']
    colors = sns.color_palette(None, 2)
    plot_roc(ax, y_test, y_pred_score, None, color=colors[0], label='P-NET (70K params)')

    y_test = df_dense['y']
    y_pred_score = df_dense['pred_scores']
    plot_roc(ax, y_test, y_pred_score, None, color=colors[1], label='Dense (14M params)')
    plt.legend(loc="lower right", fontsize=fontsize, framealpha=0.0)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=fontsize)
    ax.set_yticklabels(ax.get_yticks(), size=fontsize)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    
def plot_pnet_vs_dense(axis):
    pvalus_list = []
    for i, (c, l) in enumerate(zip(cols, labels)):
        pvalues = plot_pnet_vs_densenet(axis[i], c, label=l, plot_ratio=False)
        pvalus_list.append(pvalues)

    df = pd.DataFrame(pvalus_list, index=labels)
    filename = os.path.join(saving_dir, 'pnet_vs_dense_sameweights_tests.csv')
    df.round(3).to_csv(os.path.join(saving_dir, filename))
    
    
fontsize = 6  # legends, axis
fontproperties = {'family': 'Arial', 'weight': 'normal', 'size': 6}

saving_dir = os.path.join(PLOTS_PATH, 'extended_figures')
if not os.path.exists(saving_dir):
    os.makedirs(saving_dir)


def run():
    base_dir = PROSTATE_LOG_PATH
    models_base_dir = os.path.join(base_dir, 'other_ml_models/onesplit_ml_test')
        
    fig = plt.figure(constrained_layout=False, figsize=(7.2, 9.72))

    spec2 = gridspec.GridSpec(ncols=2, nrows=4, figure=fig)
    ax1 = fig.add_subplot(spec2[0, 0])
    ax2 = fig.add_subplot(spec2[0, 1])
    ax3 = fig.add_subplot(spec2[1, 0])
    ax4 = fig.add_subplot(spec2[1, 1])
    ax5 = fig.add_subplot(spec2[2, 0])
    ax6 = fig.add_subplot(spec2[2, 1])

    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.99, wspace=0.7, hspace=0.5)
    saving_filename = os.path.join(saving_dir, 'pnet_densenet_performance_comparision.png')

    plot_pnet_vs_dense(axis=[ax1, ax2, ax3, ax4, ax5])

    plot_pnet_vs_dense_same_arch(ax6)
    plt.savefig(saving_filename, dpi=300)


if __name__ == '__main__':
    run()