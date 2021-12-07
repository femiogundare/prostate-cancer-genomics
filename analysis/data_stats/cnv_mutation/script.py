# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 23:53:43 2021

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
filename = '41588_2018_78_MOESM5_ESM.xlsx'
saving_dir = os.path.join(PLOTS_PATH, 'data_stats')

if not os.path.exists(saving_dir):
    os.makedirs(saving_dir)
    
df = pd.read_excel(os.path.join(base_dir, filename), skiprows=2)

df_metastatic = df[df['Sample.Type'] == 'Metastasis'].copy()
df_primary = df[df['Sample.Type'] == 'Primary'].copy()

mutation_metastatic = df_metastatic['Mutation_count']
cnv_metastatic = df_metastatic['Fraction of genome altered']

mutation_primary = df_primary['Mutation_count']
cnv_primary = df_primary['Fraction of genome altered']


fig = plt.figure()
fig.set_size_inches(12, 8)
fig.suptitle('Mutation against Copy-number alteration', fontsize=18)

plt.subplot(3, 1, 1)
plt.plot(np.log(1+mutation_metastatic), cnv_metastatic, 'r.')
plt.title('Metastatic')
plt.ylabel('Copy-number alteration', fontsize=10)
plt.xlim((0, 8))

plt.subplot(3, 1, 2)
plt.plot(np.log(1+mutation_primary),cnv_primary , 'b.')
plt.title('Primary')
plt.ylabel('Copy-number alteration', fontsize=10)
plt.xlim((0, 8))

plt.subplot(3, 1, 3)
plt.scatter(np.log(1+mutation_metastatic), cnv_metastatic, edgecolors ='r', facecolors='none')
plt.scatter(np.log(1+mutation_primary), cnv_primary, edgecolors= 'b', facecolors='none')
plt.xlim((0, 8))
plt.title('Primary and Metastatic')
plt.xlabel('log(1+ # mutations)', fontsize=18)
plt.ylabel('Copy-number alteration', fontsize=10)
plt.legend(['Primary', 'Metastatic'])
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
filename = os.path.join(saving_dir, 'cnv_mutation.png')
plt.savefig(filename)
plt.show()