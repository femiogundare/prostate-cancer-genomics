# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 01:39:30 2021

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
files.append(dict(Model='Fusion', file=os.path.join(base_dir, 'onsplit_average_reg_10_tanh_large_testing_fusion')))
files.append(dict(Model='no-Fusion', file=os.path.join(base_dir, 'onsplit_average_reg_10_tanh_large_testing_fusion_zero')))
files.append(dict(Model='Fusion (genes)', file=os.path.join(base_dir, 'onsplit_average_reg_10_tanh_large_testing_inner_fusion_genes')))
dirs_df = pd.DataFrame(files)



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
    
    
def plot_auc(ax, model_dict):
    plot_auc_all(model_dict, ax)
    ax.set_xticklabels(ax.get_xticks(), fontsize=fontsize)
    ax.set_yticklabels(ax.get_yticks(), fontsize=fontsize)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.legend(loc="lower right", fontsize=fontsize, framealpha=0.0)
    
    
def sort_dict(all_models_dict):
    sorted_dict = {}
    for i, k in enumerate(all_models_dict.keys()):
        df = all_models_dict[k]
        y_test = df['y']
        y_pred_score = df['pred_scores']
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_score, pos_label=1)
        average_auc = metrics.auc(fpr, tpr)
        sorted_dict[k] = average_auc
        print('Model {} , auc= {}'.format(k, average_auc))

    sorted_dict = sorted(sorted_dict.items(), key=lambda kv: kv[1], reverse=True)
    sorted_dict = collections.OrderedDict(sorted_dict)
    
    return sorted_dict


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
        
        
def plot_auc_bootstrap(all_models_dict, ax):
    n = len(all_models_dict.keys())
    colors = sns.color_palette(None, n)

    all_scores = []
    names = []
    xs = []
    avg_scores = []
    for i, k in enumerate(all_models_dict.keys()):
        df = all_models_dict[k]
        y_test = df['y']
        y_pred_score = df['pred_scores']
        score, ci_lower, ci_upper, scores = score_ci(y_test, y_pred_score, score_fun=metrics.roc_auc_score,
                                                     n_bootstraps=2000, seed=123)
        all_scores.append(scores)
        names.append(k)
        xs.append(np.random.normal(i + 1, 0.04, len(scores)))
        avg_scores.append(score)

    all_scores = [x for _, x in sorted(zip(avg_scores, all_scores))]
    names = [x for _, x in sorted(zip(avg_scores, names))]

    flierprops = dict(marker='o', markersize=1, alpha=0.7)
    ax.boxplot(all_scores, labels=names, flierprops=flierprops)
    ngroup = len(all_scores)
    clevels = np.linspace(0., 1., ngroup)
    for i, (x, val, clevel) in enumerate(zip(xs, all_scores, clevels)):
        ax.scatter(x, val, marker='.', color=colors[i], alpha=0.1, linewidths=0.2, s=1)
        
        
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


def get_contributions(fusion='no-Fusion'):
    f = 'fs/coef_P-net_ALL_layerinputs.csv'
    base_dir = dirs_df[dirs_df.Model == fusion].file.values[0]
    f = os.path.join(base_dir, f)
    coef_df = pd.read_csv(f)
    if fusion == 'Fusion':
        coef_df.columns = ['type', 'gene', 'feature', 'coef']
    else:
        coef_df.columns = ['gene', 'feature', 'coef']
    coef_df['coef_abs'] = coef_df.coef.abs()
    coef_df.head()
    plot_df = coef_df.groupby('feature').coef_abs.sum()
    plot_df = 100 * plot_df / plot_df.sum()
    plot_df.sort_values()
    plot_df = plot_df.to_frame()
    plot_df.columns = [fusion]
    return plot_df


D_id_color = {'Amplification': [0.8784313725490196, 0.4823529411764706, 0.2235294117647059, 0.7],
              'Mutation': [0.4117647058823529, 0.7411764705882353, 0.8235294117647058, 0.7],
              'Deletion': [0.00392156862745098, 0.21568627450980393, 0.5803921568627451, 0.7],
              'Fusion (genes)': 'red',
              'Fusion (indicator)': 'yellow',
              }

mapping = {'cnv_amp': 'Amplification', 'cnv_del': 'Deletion', 'mut_important': 'Mutation',
           'fusion_genes': 'Fusion (genes)', 'fusion_indicator': 'Fusion (indicator)'}


def plot_contributions(ax):
    models = ['Fusion', 'no-Fusion', 'Fusion (genes)']

    contibution_list = []
    for m in models:
        df = get_contributions(fusion=m)
        contibution_list.append(df)

    plot_df = pd.concat(contibution_list, axis=1, sort=False)

    plot_df = plot_df.rename(index=mapping)
    plot_df.fillna(0, inplace=True)
    color = [D_id_color[i] for i in plot_df.index]

    plot_df.T.plot.bar(stacked=True, color=color, rot=0, ax=ax)
    ax.legend(framealpha=0.0, bbox_to_anchor=(.5, 1.2), loc='upper center', ncol=2, prop={'size': fontsize})

    ax.set_ylabel('Percent of relative contribution of data types (%)', fontsize=fontsize)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    
def plot_auc_bootstrap(ax, model_dict):
    plot_auc_bootstrap(model_dict, ax)
    ax.set_ylabel('AUC (bootstrap)', fontsize=fontsize)
    ax.set_ylim(0.5, 1.05)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=fontsize)
    ax.set_yticklabels(ax.get_yticks(), fontsize=fontsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    
def plot_common_top(ax, dirs_df):
    def read_feature_ranks(dirs_df):
        layers = ['h0', 'h1', 'h2', 'h3', 'h4', 'h5']
        model_dict = {}
        for l in layers:
            coef_df_list = []
            keys = []
            for i, row in dirs_df.iterrows():
                dir_ = row.file
                model = row.Model
                dir_ = os.path.join(dir_, 'fs')
                f = 'coef_P-net_ALL_layer{}.csv'.format(l)
                coef_file = os.path.join(dir_, f)
                coef_df = pd.read_csv(coef_file, index_col=0)
                coef_df.columns = [model]
                coef_df_list.append(coef_df)
                keys.append(model)
            coef_df = pd.concat(coef_df_list, axis=1)
            model_dict[l] = coef_df

        return model_dict

    coef_df_dict = read_feature_ranks(dirs_df)

    n = 20
    common_list = []
    common_list_genes = []
    layers = ['h0', 'h1', 'h2', 'h3', 'h4', 'h5']
    for l in layers:
        ranked = coef_df_dict[l].abs().rank(ascending=False)
        top_fusions_genes = ranked['Fusion (genes)'].nsmallest(n).index
        top_fusions = ranked['Fusion'].nsmallest(n).index
        top_nofusions = ranked['no-Fusion'].nsmallest(n).index

        c = len(set(top_fusions).intersection(top_nofusions))
        common_list.append(100. * c / float(n))

        c = len(set(top_fusions_genes).intersection(top_nofusions))
        common_list_genes.append(100. * c / float(n))

    ax.plot(common_list, '-.')
    ax.plot(common_list_genes, '-.')
    ax.set_ylim(0, 100)
    ax.set_ylabel('Percent of common nodes (%)', fontproperties)
    ax.set_xlabel('Layers', fontproperties)
    ax.legend(['Fusion', 'Fusion (genes)'], fontsize=fontsize, framealpha=0)

    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_xticklabels(layers)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    
def plot_fusion_indicator_ranking(ax, dirs_df):
    f = 'fs/coef_P-net_ALL_layerinputs.csv'
    base_dir = dirs_df[dirs_df.Model == 'Fusion'].file.values[0]
    f = os.path.join(base_dir, f)

    coef_df = pd.read_csv(f)
    coef_df.columns = ['type', 'gene', 'feature', 'coef']
    coef_df['coef_abs'] = coef_df.coef.abs()
    plot_df = coef_df.groupby('feature').coef_abs.sum()
    plot_df = 100 * plot_df / plot_df.sum()
    plot_df.sort_values()

    col = 'coef_abs'
    importance = coef_df.sort_values(col, ascending=False)
    importance['rank'] = range(1, len(importance) + 1)
    importance_log = np.log(importance[col].values + 1)

    ax.plot(importance_log, ".", linewidth=1.)
    ax.set_ylabel('Log (importance score +1)', fontsize=fontsize)

    ind = importance.feature == 'fusion_indicator'
    y = importance_log[ind][0]
    x = importance.loc[ind, 'rank'].values[0]
    ax.annotate('Fusion indicator', (x, y),
                xycoords='data',
                fontsize=8,
                bbox=dict(boxstyle="round", fc="none", ec="gray"),
                xytext=(60, 40), textcoords='offset points', ha='center',
                arrowprops=dict(arrowstyle="->"))

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
    ax2 = fig.add_subplot(spec2[0, 1])
    ax3 = fig.add_subplot(spec2[1, 0])
    ax4 = fig.add_subplot(spec2[1, 1])
    ax5 = fig.add_subplot(spec2[2, 0])

    plot_auc(ax1, model_dict)
    plot_auc_bootstrap(ax2, model_dict)
    plot_fusion_indicator_ranking(ax3, dirs_df)
    plot_contributions(ax4)
    plot_common_top(ax5, dirs_df)

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    fig.tight_layout()

    plt.gcf().subplots_adjust(left=0.1, right=0.9, bottom=0.2, wspace=0.7, hspace=0.4)
    filename = os.path.join(saving_dir, 'fusions.png')
    plt.savefig(filename, dpi=300)


if __name__ == '__main__':
    run()