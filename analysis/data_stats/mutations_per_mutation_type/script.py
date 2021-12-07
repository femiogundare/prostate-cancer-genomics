# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 08:26:34 2021

@author: femiogundare
"""

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[3]))

from config import PROSTATE_DATA_PATH, PLOTS_PATH


base_dir = os.path.join(PROSTATE_DATA_PATH, 'raw_data')
filename = '41588_2018_78_MOESM4_ESM.txt'
saving_dir = os.path.join(PLOTS_PATH, 'data_stats')

df = pd.read_csv(os.path.join(base_dir, filename), sep='\t', low_memory=False, skiprows=1)

print(df['type'].value_counts())
counts = np.log(1 + df['type'].value_counts())
fig = plt.gcf()
fig.clf()
ax = plt.subplot(111)

plt.bar(np.arange(len(counts.index)), counts.values, color=(0.2, 0.4, 0.6, 0.6))
plt.xticks(np.arange(len(counts.index)), counts.index, rotation='vertical')

for spine in plt.gca().spines.values():
    spine.set_visible(False)

plt.gcf().subplots_adjust(bottom=0.4)
plt.title('Number of mutations per mutation type', size=16)
plt.xlabel('Mutation type', fontsize=18)
plt.ylabel('Log (1+mutation count)', fontsize=14)
ax.yaxis.set_ticks_position('none')
ax.xaxis.set_ticks_position('none')
ax.grid(axis='y', color='white', linestyle='-')
filename = os.path.join(saving_dir, 'mutations_per_mutation_type.png')
plt.savefig(filename)
plt.show()