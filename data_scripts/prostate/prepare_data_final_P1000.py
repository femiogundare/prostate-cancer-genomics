# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 18:10:57 2021

@author: femiogundare
"""

import os
import pandas as pd
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from config import PROSTATE_DATA_PATH

data_dir = os.path.join(PROSTATE_DATA_PATH, 'raw_data')
processed_dir = os.path.join(PROSTATE_DATA_PATH, 'processed')


def build_design_matrix_crosstable():
    print('Building mutations...')
    filename = '41588_2018_78_MOESM4_ESM.txt'
    id_col = 'Tumor_Sample_Barcode'
    
    df = pd.read_csv(os.path.join(data_dir, filename), sep='\t', low_memory=False, skiprows=1)
    print('Mutation distribution')
    print(df['Variant_Classification'].value_counts())
    
    if filter_silent_muts:
        #filter silent mutations
        df = df[df['Variant_Classification'] != 'Silent'].copy()
    if filter_missense_muts:
        #filter missense mutations
        df = df[df['Variant_Classification'] != 'Missense_Mutation'].copy()
    if filter_introns_muts:
        #filter intron mutations
        df = df[df['Variant_Classification'] != 'Intron'].copy()
        
    
    if keep_important_only:
        #keep important mutations only
        exclude = ['Silent', 'Intron', "3\'UTR", "5\'UTR", 'RNA', 'lincRNA']
        df = df[~df['Variant_Classification'].isin(exclude)].copy()
    if truncating_only:
        include = ['Nonsense_Mutation', 'Frame_Shift_Del', 'Frame_Shift_Ins']
        df = df[df['Variant_Classification'].isin(include)].copy()
        
    df_table = pd.pivot_table(data=df, index=id_col, columns='Hugo_Symbol', values='Variant_Classification',
                              aggfunc='count')
    df_table = df_table.fillna(0)
    total_number_of_mutations = df_table.sum().sum()

    number_samples = df_table.shape[0]
    print('Number of mutations', total_number_of_mutations, total_number_of_mutations // (number_samples + 0.0))
    
    filename = os.path.join(processed_dir, 'P1000_final_analysis_set_cross_' + ext + '.csv')
    df_table.to_csv(filename)
    
    
    
def build_response():
    print('Building response...')
    filename = '41588_2018_78_MOESM5_ESM.xlsx'
    df = pd.read_excel(os.path.join(data_dir, filename), sheet_name='Supplementary_Table3.txt', skiprows=2)
    response = pd.DataFrame()
    response['id'] = df['Patient.ID']
    response['response'] = df['Sample.Type']
    response['response'] = response['response'].replace('Metastasis', 1)
    response['response'] = response['response'].replace('Primary', 0)
    response = response.drop_duplicates()
    response.to_csv(os.path.join(processed_dir, 'response_paper.csv'), index=False)
    
    
    
def build_cnv():
    print('Building copy number variants...')
    filename = '41588_2018_78_MOESM10_ESM.txt'
    df = pd.read_csv(os.path.join(data_dir, filename), sep='\t', low_memory=False, skiprows=1, index_col=0)
    df = df.T
    df = df.fillna(0.)
    filename = os.path.join(processed_dir, 'P1000_data_CNA_paper.csv')
    df.to_csv(filename)
    
    
    
def build_cnv_burden():
    print('Building copy number variants burden')
    filename = '41588_2018_78_MOESM5_ESM.xlsx'
    df = pd.read_excel(os.path.join(data_dir, filename), skiprows=2, index_col=1)
    cnv = df['Fraction of genome altered']
    filename = os.path.join(processed_dir, 'P1000_data_CNA_burden.csv')
    cnv.to_frame().to_csv(filename)
    
    
    
filter_silent_muts = False
filter_missense_muts = False
filter_introns_muts = False
keep_important_only = True
truncating_only = False

ext = ""
if keep_important_only:
    ext = 'important_only'

if truncating_only:
    ext = 'truncating_only'

if filter_silent_muts:
    ext = "_no_silent"

if filter_missense_muts:
    ext = ext + "_no_missense"

if filter_introns_muts:
    ext = ext + "_no_introns"

build_design_matrix_crosstable()
build_cnv()
build_response()
build_cnv_burden()
print('Done')