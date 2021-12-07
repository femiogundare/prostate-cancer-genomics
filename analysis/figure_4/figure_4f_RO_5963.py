# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 21:49:43 2021

@author: femiogundare
"""

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
from scipy.optimize import curve_fit
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from config import PROSTATE_DATA_PATH, PLOTS_PATH


def sigmoid(x, L, k, x0):
    b = 0.
    y = L / (1 + np.exp(-k * (x - x0))) + b
    return (y)


def run():
    saving_dir = os.path.join(PLOTS_PATH, 'figure4')
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)
        
    fig, ax = plt.subplots(nrows=1, ncols=1, sharey=True, figsize=(5, 4), dpi=200)
    filename = os.path.join(PROSTATE_DATA_PATH, 'functional/Data S3 CRISPR and RO-5963 results.xlsx')
    df = pd.read_excel(filename, sheet_name='RO drug curves', header=[0, 1], index_col=0)
    cols = ['LNCaP'] * 6 + ['PC3'] * 6 + ['DU145'] * 6

    exps = ['LNCaP', 'PC3', 'DU145']
    colors = {'LNCaP': 'maroon', 'PC3': '#577399', 'DU145': 'orange'}

    X = df.index.values
    legend_labels = []
    legnds = []
    for i, exp in enumerate(exps):
        legend_labels.append(exp)
        df_exp = df[exp].copy()
        stdv = df_exp.std(axis=1)
        mean = df_exp.mean(axis=1)
        ydata = df_exp.values.flatten()
        xdata = np.repeat(X, 6)
        p0 = [1.0, -1., -.7]
        popt, pcov = curve_fit(sigmoid, xdata, ydata, p0, method='dogbox', maxfev=60000)
        plt.errorbar(X, mean, yerr=stdv, fmt='o', ms=5, color=colors[exp], alpha=0.75, capsize=3, label=exp)
        x2 = np.linspace((min(xdata), max(xdata)), 10)
        y2 = sigmoid(x2, *popt)
        plt.plot(x2, y2, color=colors[exp], alpha=0.75, linewidth=2)

        legnds.append(mpatches.Patch(color=colors[exp], label=exp))

    plt.xscale('log')
    plt.ylim((-.6, 1.6))
    plt.subplots_adjust(left=0.2, bottom=0.2, right=0.8)
    plt.ylabel('Relative Viability', fontdict=dict(family='Arial', weight='bold', fontsize=14))
    plt.xlabel('RO-5963 (\u03bcM)', fontdict=dict(family='Arial', weight='bold', fontsize=14))
    ax.spines['bottom'].set_position(('data', 0.))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.yticks([-0.5, 0.0, 0.5, 1.0, 1.5])
    plt.xlim((.02, 120))

    ax.legend(handles=legnds, bbox_to_anchor=(.9, 1.), framealpha=0.0)
    plt.savefig(os.path.join(saving_dir, 'RO-5963.png'))


if __name__ == "__main__":
    run()