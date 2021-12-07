# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 22:01:34 2021

@author: femiogundare
"""

import os
import pandas as pd
from matplotlib import gridspec
from adjustText import adjust_text
from matplotlib import pyplot as plt
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from config import PROSTATE_DATA_PATH, PLOTS_PATH


def display_plot():
    filename = os.path.join(PROSTATE_DATA_PATH, 'supporting_data/Z score list in all conditions.xlsx')
    df = pd.read_excel(filename)
    df = df.set_index('gene symbol')
    df.head()
    df = df.groupby(by=df.index).max().sort_values('Z-LFC AVERAGE Enzalutimide')

    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 9, 1])

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])

    x = range(df.shape[0])

    ax1.plot(x, df['Z-LFC AVERAGE Enzalutimide'], '.')
    ax2.plot(x, df['Z-LFC AVERAGE Enzalutimide'], '.')
    ax3.plot(x, df['Z-LFC AVERAGE Enzalutimide'], '.')

    ax3.set_ylim(-15.5, -10)
    ax2.set_ylim(-6, 6)
    ax1.set_ylim(10, 15)

    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax2.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    #
    ax1.yaxis.tick_left()
    ax1.tick_params(labelright='off')
    ax2.tick_params(labelright='off')
    ax3.tick_params(labelright='off')

    ax1.set_xticklabels([])
    ax2.set_xticklabels([])
    ax3.set_xticklabels([])

    ax1.set_xticks([])
    ax2.set_xticks([])
    ax3.set_xticks([])

    ax2.set_ylabel('Z-score (CSS+enza)', fontdict=dict(weight='bold', fontsize=12))
    #interesting_genes = ['AR', 'TP53', 'PTEN', 'RB1', 'MDM4', 'FGFR1', 'MAML3', 'PDGFA', 'NOTCH1', 'EIF3E']
    interesting_genes = ['AR', 'TP53', 'PTEN', 'RB1', 'MUC16', 'TBP', 'MDM4', 'FGFR1', 'COL1A2', 'ASH2L']

    texts = []
    """
    xy_dict = dict(TP53=(30, -4),
                   PTEN=(10, 20),
                   MDM4=(-30, 4),
                   FGFR1=(-10, 20),
                   MAML3=(30, -10),
                   # PDGFA=(),
                   NOTCH1=(30, 2),
                   EIF3E=(40, -2)
    """
    
    xy_dict = dict(TP53=(30, -4),
                   PTEN=(10, 20),
                   MDM4=(-30, 4),
                   FGFR1=(-10, 20),
                   TBP=(30, -10),
                   # PDGFA=(),
                   COL1A2=(30, 2),
                   ASH2L=(40, -2),
                   #MUC16=()

                   )
    direction = [-1, 1] * 5
    x = [0, 30, -30, 0, ]
    y = [0, -2, +4, 0]
   
    for i, gene in enumerate(interesting_genes):
        if gene in df.index:
            ind = df.index.str.contains(gene)
            x = list(ind).index(True)
            y = df['Z-LFC AVERAGE Enzalutimide'].values[x]
            xytext = (direction[i] * 30, -2)
            ax2.annotate(gene, (x, y), xycoords='data', fontsize=8,
                         bbox=dict(boxstyle="round", fc="none", ec="gray"),
                         xytext=xy_dict[gene], textcoords='offset points', ha='center',
                         arrowprops=dict(arrowstyle="->"))

    adjust_text(texts)
    ax2.grid()
    plt.subplots_adjust(left=0.15)
    
    
def run():
    saving_dir = os.path.join(PLOTS_PATH, 'figure4')
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)
    display_plot()    
    filename = os.path.join(saving_dir, 'enzalutamide_resistance.png')
    plt.savefig(filename, dpi=200)
    
    
if __name__ == '__main__':
    run()