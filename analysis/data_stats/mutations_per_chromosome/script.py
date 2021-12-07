# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 01:17:20 2021

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
df = df.sort_values('chr')
counts = df['chr'].value_counts()

fig = plt.figure()
fig.set_size_inches(12, 8)
plt.bar(np.arange(len(counts.index)), counts.values, color=(0.2, 0.4, 0.6, 0.6))
plt.xticks(np.arange(len(counts.index)), counts.index)
for spine in plt.gca().spines.values():
    spine.set_visible(False)
plt.title('Number of mutations per chromosome', size=16)
plt.xlabel('Chromosome number')
plt.ylabel('Number of mutations')
filename = os.path.join(saving_dir, 'mutations_per_chromosome.png')
plt.savefig(filename)
plt.show()