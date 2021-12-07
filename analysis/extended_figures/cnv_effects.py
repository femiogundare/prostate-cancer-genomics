# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 04:19:34 2021

@author: femiogundare
"""

import os
import collections
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from sklearn import metrics
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from config import PROSTATE_LOG_PATH, PLOTS_PATH
from utilities.stats_utils import score_ci

base_dir = os.path.join(PROSTATE_LOG_PATH, 'review/fusion')

files = []
files.append(dict(Model='Fusion', file=os.path.join(base_dir, 'onesplit_average_reg_10_tanh_test_fusion')))
files.append(dict(Model='no-Fusion', file=os.path.join(base_dir, 'onesplit_average_reg_10_tanh_test_fusion_zero')))
files.append(dict(Model='Fusion (genes)', file=os.path.join(base_dir, 'onesplit_average_reg_10_tanh_test_inner_fusion_genes')))
dirs_df = pd.DataFrame(files)


def sort_dict(all_models_dict):
    sorted_dict = {}
    for i, k in enumerate(all_models_dict.keys()):
        df = all_models_dict[k]
        y_test = df['y']
        y_pred_score = df['pred_scores']
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_score, pos_label=1)
        average_auc = metrics.auc(fpr, tpr)
        sorted_dict[k] = average_auc
        print('model {} , auc= {}'.format(k, average_auc))

    sorted_dict = sorted(sorted_dict.items(), key=lambda kv: kv[1], reverse=True)
    sorted_dict = collections.OrderedDict(sorted_dict)
    return sorted_dict


def plot_stability(ax):
    def get_stability_index(df, n_features):
        n = len(df.columns)
        pairs = []
        for i in range(n):
            for j in np.arange(i, n):
                if i != j:
                    pairs.append((i, j))
        overlaps = []
        for pair in pairs:
            first, second = pair
            a = df.iloc[:, first].nlargest(n_features)
            b = df.iloc[:, second].nlargest(n_features)
            overlap = len(set(a.index).intersection(b.index))
            overlaps.append(overlap)
        avg_overlap = np.mean(overlaps) / n_features
        return avg_overlap

    def plot_stability_(ax, model_name, n_features):
        filename = model_name
        df = pd.read_csv(filename, index_col=[0, 1])
        stability_indeces = []
        for f in n_features:
            stability_index = get_stability_index(df, f)
            stability_indeces.append(stability_index)

        ax.plot(n_features, stability_indeces, '*-', linewidth=0.5)
        ax.set_ylabel('Stability index', fontsize=fontsize)
        ax.set_xlabel('Number of top features', fontsize=fontsize)

    n_features = [200, 100, 50, 20, 10]
    base_dir1 = os.path.join(PROSTATE_LOG_PATH, 'review/9single_copy/crossvalidation_average_reg_10_tanh_single_copy/fs')
    base_dir2 = os.path.join(PROSTATE_LOG_PATH, 'pnet/crossvalidation_average_reg_10_tanh/fs')
    files = []
    f = os.path.join(base_dir1, 'coef.csv')
    files.append(f)
    f = os.path.join(base_dir2, 'coef.csv')
    files.append(f)
    models = ['single copy', 'two copies']

    for m in files:
        plot_stability_(ax, m, n_features)
    ax.legend([m.replace('.csv', '') for m in models], fontsize=fontsize)
    ax.set_xlabel('Number of top features', fontproperties)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    
def read_predictions(dirs_df):
    model_dict = {}
    for i, row in dirs_df.iterrows():
        dir_ = row.file
        model = row.Model
        dir_ = os.path.join(base_dir, dir_)
        prediction_file = os.path.join(dir_, 'P-net_ALL_testing.csv')
        pred_df = pd.read_csv(prediction_file)
        model_dict[model] = pred_df
    return model_dict
    
    
def plot_roc(ax, y_test, y_pred_score, save_dir, color, label=''):
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_score, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    symbol = '-'
    ax.plot(fpr, tpr, symbol, label=label + ' (%0.3f)' % roc_auc, linewidth=1, color=color)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontproperties)
    ax.set_ylabel('True Positive Rate', fontproperties)
    
    
def plot_auc_all(all_models_dict, ax):
    n = len(all_models_dict.keys())
    colors = sns.color_palette(None, n)

    sorted_dict = sort_dict(all_models_dict)
    for i, k in enumerate(sorted_dict.keys()):
        print('Model {} , auc= {}'.format(k, sorted_dict[k]))
        df = all_models_dict[k]
        y_test = df['y']
        y_pred_score = df['pred_scores']
        plot_roc(ax, y_test, y_pred_score, None, color=colors[i], label=k)
        
        
def compare_auc_cnv_def(ax):
    base_dir_single_copy = os.path.join(PROSTATE_LOG_PATH, 'review/9single_copy')
    base_dir_two_copies = os.path.join(PROSTATE_LOG_PATH, 'pnet')

    files = []
    files.append(dict(Model='single copy',
                      file=os.path.join(base_dir_single_copy, 'onsplit_average_reg_10_tanh_large_testing_single_copy')))
    files.append(dict(Model='two copies', file=os.path.join(base_dir_two_copies, 'onsplit_average_reg_10_tanh_large_testing')))
    dirs_df = pd.DataFrame(files)

    model_dict = read_predictions(dirs_df)

    plot_auc_all(model_dict, ax)
    ax.legend(loc="lower right", fontsize=fontsize, framealpha=0.0)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    
fontsize = 5  # legends, axis
fontproperties = {'family': 'Arial', 'weight': 'normal', 'size': 6}
current_dir = os.path.basename(os.path.dirname(__file__))
saving_dir = os.path.join(PLOTS_PATH, 'extended_figures')
if not os.path.exists(saving_dir):
    os.makedirs(saving_dir)
        
model_dict = read_predictions(dirs_df)


def run():
    fig = plt.figure(constrained_layout=False, figsize=(7.2, 9.72))
    spec2 = gridspec.GridSpec(ncols=2, nrows=4, figure=fig)
    ax1 = fig.add_subplot(spec2[0, 0])
    ax3 = fig.add_subplot(spec2[1, 0])

    plot_stability(ax3)
    compare_auc_cnv_def(ax1)

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    fig.tight_layout()

    plt.gcf().subplots_adjust(left=0.1, right=0.9, bottom=0.2, wspace=0.7, hspace=0.4)
    filename = os.path.join(saving_dir, 'figure_ed5_cnv.png')
    plt.savefig(filename, dpi=300)


if __name__ == '__main__':
    run()