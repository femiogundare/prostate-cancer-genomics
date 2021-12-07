# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 22:14:35 2021

@author: femiogundare
"""

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from config import PROSTATE_DATA_PATH, PLOTS_PATH


def label_diff(ax, i, j, text, X, Y, stdv, yfactor=1.2):
    x = X[i] + (X[j] - X[i]) / 3.

    y = yfactor * max(Y[i] + stdv[i], Y[j] + stdv[j])
    y_text = y + 0.04
    dx = abs(X[i] - X[j])

    ax.annotate(text, xy=(x, y_text), zorder=10, size=8)
    plt.hlines(y=y, xmax=X[j], xmin=X[i], linewidth=1)
    plt.vlines(x=X[j], ymax=y, ymin=y - 0.06, linewidth=1)
    plt.vlines(x=X[i], ymax=y, ymin=y - 0.06, linewidth=1)
    
    
def run():
    saving_dir = os.path.join(PLOTS_PATH, 'figure4')
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)
        
    fig, ax = plt.subplots(nrows=1, ncols=1, sharey=True, figsize=(3.5, 4), dpi=200)

    filename = os.path.join(PROSTATE_DATA_PATH, 'functional/092420 Data.xlsx')
    df = pd.read_excel(filename, sheet_name='sgRNA Data')
    cols = ['sgGFP', 'sgMDM4-1', 'sgMDM4-2']
    df = df[cols]
    mean_impf = df.mean()
    sem_impf = df.std()
    yerr_pos = sem_impf.copy()

    D_id_color = ['gainsboro', 'maroon', 'maroon']
    ind = np.arange(3)
    plt.bar(ind, mean_impf, yerr=[(0, 0, 0), yerr_pos], color=D_id_color, ecolor='black',
            tick_label=cols, align='center')
    plt.ylabel('Normalized Cell Count', fontdict=dict(family='Arial', weight='bold', fontsize=14))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.subplots_adjust(left=0.20)
    ax.set_ylim((0, 1.65))
    # Call the function
    label_diff(ax, 0, 1, 'p<0.0001', ind, mean_impf, sem_impf, yfactor=1.2)
    label_diff(ax, 0, 2, 'p<0.0001', ind, mean_impf, sem_impf, yfactor=1.4)
    label_diff(ax, 1, 2, 'n.s.', ind, mean_impf, sem_impf, yfactor=1.3)

    # df.T.plot.bar(stacked=False)
    filename = os.path.join(saving_dir, 'crispr_cas9.png')
    plt.savefig(filename)
    
    
if __name__ == "__main__":
    run()