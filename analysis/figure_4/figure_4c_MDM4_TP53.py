# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 22:57:08 2021

@author: Dell
"""

from os.path import join
import os
import numpy as np
from matplotlib import pyplot as plt
from upsetplot import from_memberships
from upsetplot import plot
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from config import PROSTATE_DATA_PATH, PLOTS_PATH
#from setup import saving_dir


def run():
    saving_dir = os.path.join(PLOTS_PATH, 'figure4')
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)
        
    fig = plt.figure(figsize=(8, 8))
    data = np.array([795., 27., 182., 7.])
    plt.rcParams.update({'font.size': 14})
    example = from_memberships(
        [
            [' TP53 WT', ' MDM4 WT'],
            [' TP53 WT', ' MDM4 amp.'],
            [' TP53 mutant', ' MDM4 WT'],
            [' TP53 mutant', ' MDM4 amp.']],
        data=data
    )
    intersections, matrix, shading, totals = plot(example, fig=fig, with_lines=True, show_counts=True, element_size=50)
    plt.ylabel('Number of patients', fontdict=dict(weight='bold', fontsize=16))

    filename = join(saving_dir, 'upset_MDM4_TP53.png')
    plt.savefig(filename)


if __name__ == "__main__":
    run()
