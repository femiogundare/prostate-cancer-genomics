# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 18:51:49 2021

@author: femiogundare
"""

import os
import pandas as pd
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from config import PROSTATE_DATA_PATH, GENE_PATH

data_dir = os.path.join(PROSTATE_DATA_PATH, 'external_validaton')
processed_dir = os.path.join(PROSTATE_DATA_PATH, 'processed')


if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    
    
def get_design_matrix_mutation(saving_dir):
    filename = 'somatic_v4.csv'
    df = pd.read_csv(os.path.join(saving_dir, filename), sep=',')
    design_matrix = pd.pivot_table(data=df, values='Effect', index='Pipeline_ID', columns='Gene',
                                   aggfunc='count', fill_value=None)
    return design_matrix


def get_protein_encoding_genes():
    df_protein = pd.read_csv(os.path.join(GENE_PATH, 'HUGO_genes/protein-coding_gene_with_coordinate_minimal.txt'), sep='\t',
                             index_col=0)  # 16190
    df_protein.columns = ['start', 'end', 'symbol']
    df = df_protein
    df_other = pd.read_csv(os.path.join(GENE_PATH, 'HUGO_genes/other.txt'), sep='\t')  # 112
    genes = set(list(df_other['symbol']) + list(df['symbol']))
    return genes



def build_Met500_mut():
    saving_dir = os.path.join(data_dir, 'Met500')
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)

    protein_genes = get_protein_encoding_genes()
    print('Protein_genes', len(protein_genes))
    mut = get_design_matrix_mutation(saving_dir)
    genes = set(mut.columns.values)
    common_genes = protein_genes.intersection(genes)
    print('Number of genes {}, Number of common genes {} '.format(len(genes), len(common_genes)))

    # saving mutation matrix
    mut.to_csv(os.path.join(saving_dir, 'Met500_mut_matrix.csv'))



def build_Met500_cnv():
    # processing CNV data
    saving_dir = os.path.join(data_dir, 'Met500')
    protein_genes = get_protein_encoding_genes()
    # cnv_genes= pd.read_csv(os.path.join(data_dir,'met500_cnv_unique_genes.txt'), header=None)
    cnv_genes = pd.read_csv(os.path.join(saving_dir, 'Met500_cnv.txt'), header=0, index_col=0, sep='\t')
    # cnv_genes.columns = ['genes']
    print(cnv_genes.head())
    genes = set(cnv_genes.index)
    common_genes = protein_genes.intersection(genes)
    print('Number of cnv genes {}, Number of encoding genes {}, number of comon genes {} '.format(len(genes),
                                                                                                len(protein_genes),
                                                                                                len(common_genes)))
    
    
    
def build_PRAD():
    saving_dir = os.path.join(data_dir, 'PRAD')
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)

    import zipfile
    path_to_zip_file = os.path.join(saving_dir, '41586_2017_BFnature20788_MOESM324_ESM.zip')
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        extract_dir = os.path.join(saving_dir, 'nature20788-s2')
        zip_ref.extractall(extract_dir)

    path_to_zip_file = os.path.join(saving_dir, '41586_2017_BFnature20788_MOESM325_ESM.zip')
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(saving_dir)

    extract_dir = os.path.join(saving_dir, 'nature20788-s3')
    mut_filename = os.path.join(extract_dir, 'SI Data 1 filtered_variants_by_patient.tsv')

    mut_df = pd.read_csv(os.path.join(saving_dir, mut_filename), sep='\t')

    CPCG_cols = [c for c in mut_df.columns.values if c.startswith('CPCG')]

    exclude = ['upstream', 'downstream', 'intergenic']  # 'intronic', 'ncRNA_intronic'
    cpcg_mutations = mut_df.loc[~mut_df.Location.isin(exclude), ['Gene'] + CPCG_cols]
    xx = cpcg_mutations.groupby('Gene').sum()
    xx.T.to_csv(os.path.join(saving_dir, 'mut_matrix.csv'))

    # cnv
    extract_dir = os.path.join(saving_dir, 'nature20788-s2')
    cnv_filename = os.path.join(extract_dir, 'Supplementary Table 02 - Per-gene CNA analyses.xlsx')
    cna_df = pd.read_excel(os.path.join(saving_dir, cnv_filename))
    cna_df_matrix = cna_df.loc[:, ['Symbol'] + CPCG_cols]
    yy = cna_df_matrix.groupby('Symbol').max()
    yy.T.to_csv(os.path.join(saving_dir, 'cnv_matrix.csv'))
    
    
    
    
build_Met500_mut()
build_Met500_cnv()
build_PRAD()
print('Done')