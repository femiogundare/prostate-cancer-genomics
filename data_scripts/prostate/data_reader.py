# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 12:42:49 2021

@author: femiogundare
"""

import os
import numpy as np
import pandas as pd
import logging
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from config import DATA_PATH, PROSTATE_DATA_PATH


PROCESSED_PATH = os.path.join(PROSTATE_DATA_PATH, 'processed')
cnv_filename = 'P1000_data_CNA_paper.csv'    #copy number variants
response_filename = 'response_paper.csv'    #tumor responses
gene_important_mutations_only = 'P1000_final_analysis_set_cross_important_only.csv'    #mutations
gene_important_mutations_only_plus_hotspots = 'P1000_final_analysis_set_cross_important_only_plus_hotspots.csv'    #mutations
gene_hotspots = 'P1000_final_analysis_set_cross_hotspots.csv'    #mutations
gene_expression = 'P1000_adjusted_TPM.csv'
fusions_filename = 'p1000_onco_ets_fusions.csv'
cnv_burden_filename = 'P1000_data_CNA_burden.csv'    #copy number variants burden
fusions_genes_filename = 'fusion_genes.csv'

cached_data = {}

def load_data(filename, selected_genes=None):
    """
    Loads data from a file.
    
    Args:
        filename : name of file.
        selected_genes : an array of genes.
        
    Returns:
        A dataframe, an array of tumor samples, an array of patients' responses to tumors, an array of genes. 
    """
    filename = os.path.join(PROCESSED_PATH, filename)
    logging.info('Loading data from {}...'.format(filename))
    
    if filename in cached_data:
        logging.info('Loading from memory cached_data...')
        data = cached_data[filename]
    else:
        data = pd.read_csv(filename, index_col=0)
        cached_data[filename] = data
    
    if 'response' in cached_data:
        logging.info('Loading from memory cached_data...')
        labels = cached_data['response']
    else:
        logging.info('Loading response from {}...'.format(os.path.join(PROCESSED_PATH, response_filename)))
        labels = pd.read_csv(os.path.join(PROCESSED_PATH, response_filename))
        labels = labels.set_index('id')
        cached_data['response'] = labels
        
    # Join data with the labels
    all_data = data.join(labels)
    all_data = all_data[~all_data['response'].isnull()]
    
    response = all_data['response']
    samples = all_data.index
    
    del all_data['response']
    df = all_data
    genes = all_data.columns
    
    if not selected_genes is None:
        intersection = set.intersection(set(genes), selected_genes)
        if len(intersection) < len(selected_genes):
            logging.warning('Some of the genes inputed do not exist in the original dataset.')
        
        df = df.loc[:, intersection]
        genes = intersection
    logging.info('Loaded data has {} samples, {} variables, and {} responses.'.format(df.shape[0], df.shape[1], response.shape[0]))
    logging.info('Number of genes {}.'.format(len(genes)))
    
    return df, samples, response, genes


def load_TMB(filename):
    df, samples, response, genes = load_data(filename)
    x = np.sum(df, axis=1)
    x = np.array(x)
    x = np.log(1.0 + x)
    n = x.shape[0]
    samples = np.array(samples)
    response = response.values.reshape((n, 1))
    column = np.array(['TMB'])
    return x, samples, response, column


def load_CNV_burden(filename):
    """
    Loads a file containing copy number variant data.
    
    Returns:
        A dataframe, an array of tumor samples, an array of patients' responses to tumors, an array of genes.
    """
    df, samples, response, genes = load_data(filename)
    x = np.sum(df, axis=1)
    x = np.array(x)
    x = np.log(1.0 + x)
    n = x.shape[0]
    samples = np.array(samples)
    response = response.values.reshape((n, 1))
    column = np.array(['TMB'])
    return x, samples, response, column


def load_data_type(data_type='gene', cnv_levels=5, cnv_filter_single_event=True, mutation_binary=False, selected_genes=None):
    """
    Loads any type of genomic data.
    
    Args:
        data_type (str): type of data. Possible values include gene_expression, fusions, fusion_genes, important_mutations, etc.
        cnv_levels (int): level of copy number variation.
        cnv_filter_single_event (bool): Whether or not to exclude single-copy amplification and deletions.
        mutation_binary (bool): Whether or not mutation is binary.
        selected_genes (str): name of .csv file containing a unique set of genes to be selected from the genomic dataset.
        
    Returns:
        A dataframe, an array of tumor samples, an array of patients' responses to tumors, an array of genes.
    """
    logging.info('Loading {}...'.format(data_type))
    
    if data_type == 'TMB':
        x, samples, response, genes = load_TMB(gene_important_mutations_only)
        
    if data_type == 'important_mutations':
        df, samples, response, genes = load_data(gene_important_mutations_only, selected_genes)
        x = df
        if mutation_binary:
            logging.info('Mutation is binary (True).')
            x[x > 1.0] = 1.0
            
    if data_type == 'important_mutations_plus_hotspots':
        df, samples, response, genes = load_data(gene_important_mutations_only_plus_hotspots, selected_genes)
        x = df
       
    if data_type == 'mutation_hotspots':
        df, samples, response, genes = load_data(gene_hotspots, selected_genes)
        x = df
        
    if data_type == 'cnv':
        df, samples, response, genes = load_data(cnv_filename, selected_genes)
        x = df
        if cnv_levels == 3:
            logging.info('CNV Levels = {}'.format(cnv_levels))
            # Remove single amplication and single deletion, as they are usually noisy
            if cnv_filter_single_event:
                x[x == -1.0] = 0.0
                x[x == -2.0] = 1.0
                x[x == 1.0] = 0.0
                x[x == 2.0] = 1.0
            else:
                x[x < 0.0] = -1.
                x[x > 0.0] = 1.
    
    if data_type == 'cnv_deletion':
        df, samples, response, genes = load_data(cnv_filename, selected_genes)
        x = df
        x[x >= 0.0] = 0.0
        if cnv_levels == 3:
            logging.info('CNV Levels = {}'.format(cnv_levels))
            if cnv_filter_single_event:
                x[x == -1.0] = 0.0
                x[x == -2.0] = 1.0
            else:
                x[x < 0.0] = 1.0
        else:
            x[x == -1.0] = 0.5
            x[x == -2.0] = 1.0
    
    if data_type == 'cnv_amplification':
        df, samples, response, genes = load_data(cnv_filename, selected_genes)
        x = df
        x[x <= 0.0] = 0.0
        if cnv_levels == 3:
            logging.info('CNV Levels = {}'.format(cnv_levels))
            if cnv_filter_single_event:
                # Exclude single-copy amplification and deletions
                x[x == 1.0] = 0.0
                x[x == 2.0] = 1.0
            else:
                x[x > 0.0] = 1.0
                
        else:
            x[x == 1.0] = 0.5
            x[x == 2.0] = 1.0
            
            
    if data_type == 'cnv_single_deletion':
        df, samples, response, genes = load_data(cnv_filename, selected_genes)
        x = df
        x[x == -1.] = 1.0
        x[x != -1.] = 0.0
        
    if data_type == 'cnv_single_amplification':
        df, samples, response, genes = load_data(cnv_filename, selected_genes)
        x = df
        x[x == 1.] = 1.0
        x[x != 1.] = 0.0
        
    if data_type == 'cnv_high_amplification':
        df, samples, response, genes = load_data(cnv_filename, selected_genes)
        x = df
        x[x == 2.] = 1.0
        x[x != 2.] = 0.0
        
    if data_type == 'cnv_deep_deletion':
        df, samples, response, genes = load_data(cnv_filename, selected_genes)
        x = df
        x[x == -2.] = 1.0
        x[x != -2.] = 0.0
        
        
    if data_type == 'gene_expression':
        df, samples, response, genes = load_data(gene_expression, selected_genes)
        x = df
        
    if data_type == 'fusions':
        df, samples, response, genes = load_data(fusions_filename, None)
        x = df
        
    if data_type == 'cnv_burden':
        df, samples, response, genes = load_data(cnv_burden_filename, None)
        x = df
        
    if data_type == 'fusion_genes':
        df, samples, response, genes = load_data(cnv_burden_filename, selected_genes)
        x = df
        
    return x, samples, response, genes



def combine(x_list, samples_list, responses_list, genes_list, data_type_list, combine_type, use_coding_genes_only=False):
    genes_list_set = [set(list(genes)) for genes in genes_list]
    
    print('Data combine type: {}'.format(combine_type))
    
    if combine_type == 'intersection':
        genes = set.intersection(*genes_list_set)
    else:
        genes = set.union(*genes_list_set)
        
    if use_coding_genes_only:
        coding_genes_file = os.path.join(DATA_PATH, 'genes/HUGO_genes/protein-coding_gene_with_coordinate_minimal.txt')
        coding_genes_df = pd.read_csv(coding_genes_file, sep='\t', header=None)
        coding_genes_df.columns = ['chromosome', 'start', 'end', 'name']
        coding_genes = set(coding_genes_df['name'].unique())
        genes = genes.intersection(coding_genes)
        
    all_genes = list(genes)
    
    all_genes_df = pd.DataFrame(index=all_genes)
    
    df_list = []
    
    for x, samples, response, genes in zip(x_list, samples_list, responses_list, genes_list):
        df = pd.DataFrame(x, columns=genes, index=samples)
        df = df.T.join(all_genes_df, how='right')
        df =  df.T
        df = df.fillna(0)
        df_list.append(df)
        
    all_data = pd.concat(df_list, keys=data_type_list, join='inner', axis=1)
    
    # put genes on the first level and then the data type
    all_data = all_data.swaplevel(i=0, j=1, axis=1)
    
    # order the columns based on genes
    order = all_data.columns.levels[0]
    all_data = all_data.reindex(columns=order, level=0)
    
    x = all_data.values
    
    reordering_df = pd.DataFrame(index=all_data.index)
    response = reordering_df.join(response, how='left')
    
    response = response.values
    genes = all_data.columns
    samples = all_data.index
    
    logging.info('After combining, loaded data has {} samples, {} variables, and {} responses.'.format(x.shape[0], x.shape[1], response.shape[0]))
    
    return x, samples, response, genes



def split_cnv(x):
    genes = x.columns.levels[0]
    x.rename(columns={'cnv': 'CNA_amplification'}, inplace=True)
    for gene in genes:
        x[gene, 'CNA_deletion'] = x[gene, 'CNA_amplification'].replace({-1.0: 0.5, -2.0: 1.0})
        x[gene, 'CNA_amplification'] = x[gene, 'CNA_amplification'].replace({1.0: 0.5, 2.0: 1.0})
    x = x.reindex(columns=genes, level=0)
    return x



class ProstateData:
    """
    Performs a series of operations on genomic data.
    """
    def __init__(self, data_type='important_mutations', account_for_data_type=None, cnv_levels=5, cnv_filter_single_event=True,
                 mutation_binary=False, selected_genes=None, combine_type='intersection', use_coding_genes_only=False,
                 drop_AR=False, balanced_data=False, cnv_split=False, shuffle=False, selected_samples=None, training_split=0):
        """
        Parameters:
            
            data_type : str or list
                The type of data to be fed into the class. The type of data could be mutation data, copy number variant data,
                gene expression data, fusion data, etc. If a list is parsed, a loop if run to load each data type by calling
                the function, load_data_type, above. If a string is parsed, function load_data_type is called just once.
            
            account_for_data_type: list, optional
                Data types special consideration should be given to during the series operations of the class.
            
            cnv_levels: int
                Level of copy number variations.
                
            cnv_filter_single_event: bool, default True
                Whether or not to exclude single-copy amplification and deletions.  
            
            mutation_binary: bool, default False
                Whether or not mutation is binary.
                
            selected_genes: str or list, default None
                A unique set of genes to be selected from the genomic dataset. If a list is parsed, it should contain the names
                of the genes. If a string is parsed, it should be the name of a .csv file that contains a dataframe containing 
                the names of the genes in one of its columns.
                
            combine_type: str, default intersection
                Type of combination operation to be performed on a list of genes. Possible values include intersection and union.
                
            use_coding_genes_only: bool, default False
                Whether or not HUGO protein coding genes should be used in the operations.
                
            drop_AR: bool, default False
                Whether or not AR, a putative oncogene, should be dropped from the dataset.
                
            balanced_data: bool, default False
                Whether or not the data should be balanced by allowing equal representation of both classes.
                
            cnv_split: bool, default False
                Whether or not to apply the split_cnv function.
                
            shuffle: bool, default False
                Whether or not the data should be shuffled.
                
            selected_samples: str, optional
                A unique set of samples (i.e patients) to be selected from the dataset. The string parsed should be a .csv file
                that contains a dataframe containing the names of the samples in one of its columns.
            
            training_split: int
                Training dataset to be used. Values range between 0 and 19.
            
        """
        
        self.training_split = training_split
        
        if not selected_genes is None:
            if type(selected_genes) == list:
                selected_genes = selected_genes
            else:
                # Open a file that contains the desired genes
                selected_genes_file = os.path.join(DATA_PATH, 'genes', selected_genes)
                df = pd.read_csv(selected_genes_file, header=0)
                selected_genes = list(df['genes'])
        
        
        if type(data_type) == list:
            x_list = []
            samples_list = []
            responses_list = []
            genes_list = []
            
            for dtype in data_type:
                x, samples, response, genes = load_data_type(data_type=dtype, cnv_levels=cnv_levels, 
                                                             cnv_filter_single_event=cnv_filter_single_event, 
                                                             mutation_binary=mutation_binary, selected_genes=selected_genes
                                                             )
                x_list.append(x)
                samples_list.append(samples)
                responses_list.append(response)
                genes_list.append(genes)
                
            x, samples, response, genes = combine(x_list=x_list, samples_list=samples_list, responses_list=responses_list, 
                                                  genes_list=genes_list, data_type_list=data_type,
                                                  combine_type=combine_type, use_coding_genes_only=use_coding_genes_only 
                                                  )
            x = pd.DataFrame(x, columns=genes)
            
        else:
            x, samples, response, genes = load_data_type(data_type=data_type, cnv_levels=cnv_levels, 
                                                             cnv_filter_single_event=cnv_filter_single_event, 
                                                             mutation_binary=mutation_binary, selected_genes=selected_genes
                                                             )
            
            
        if drop_AR:
            data_types = x.columns.levels[1].unique()
            ind = True
            if 'cnv' in data_types:
                ind = x[('AR', 'cnv')] <= 0.0
            elif 'cnv_amplification' in data_types:
                ind = x[('AR', 'cnv_amplification')] <= 0.0
                
            if 'important_mutations' in data_types:
                ind2 = (x[('AR', 'important_mutations')] < 1.0)
                ind = ind & ind2
            
            x = x.loc[ind, ]
            response = response[ind]
            samples = samples[ind]
            
            
        if cnv_split:
            x = split_cnv(x)
            
        if type(x) == pd.DataFrame:
            x = x.values
            
        if balanced_data:
            pos_indices = np.where(response==1)[0]
            neg_indices = np.where(response==0)[0]
            
            n_pos = pos_indices.shape[0]
            n_neg = neg_indices.shape[0]
            n = min(n_pos, n_neg)
            
            pos_indices = np.random.choice(pos_indices, size=n, replace=False)
            neg_indices = np.random.choice(neg_indices, size=n, replace=False)
            
            ind = np.sort(np.concatenate([pos_indices, neg_indices]))
            
            x = x[ind, ]
            samples = samples[ind]
            response = response[ind]
            
        if shuffle:
            n = x.shape[0]
            ind = np.arange(n)
            np.random.shuffle(ind)
            x = x[ind, :]
            samples = samples[ind]
            response = response[ind, :]
            
        if account_for_data_type is not None:
            X_genomics = pd.DataFrame(x, columns=genes, index=samples)
            response_genomics = pd.DataFrame(response, columns=['response'], index=samples)
            
            X_list = []
            samples_list = []
            response_list = []
            genes_list = []
            
            for dtype in account_for_data_type:
                x_, samples_, response_, genes_ = load_data_type(data_type=dtype, cnv_levels=cnv_levels, 
                                                             cnv_filter_single_event=cnv_filter_single_event, 
                                                             mutation_binary=mutation_binary, selected_genes=selected_genes
                                                             )
                x_df = pd.DataFrame(x_, columns=genes_, index=samples_)
                X_list.append(x_df)
                samples_list.append(samples_)
                response_list.append(response_)
                genes_list.append(genes_)
                
            x_account_for = pd.concat(X_list, keys=account_for_data_type, join='inner', axis=1)
            x_all = pd.concat([X_genomics, x_account_for], keys=['genomics', 'account_for'], join='inner', axis=1)
            
            common_samples = set(samples).intersection(x_all.index)
            x_all = x_all.loc[common_samples, :]
            response = response_genomics.loc[common_samples, :]

            x = x_all.values
            samples = x_all.index
            response = response['response'].values
            genes = x_all.columns
            
            
        if selected_samples is not None:
            selected_samples_file = os.path.join(PROCESSED_PATH, selected_samples)
            selected_samples_df = pd.read_csv(selected_samples_file, header=0)
            selected_samples = list(selected_samples_df['Tumor_Sample_Barcode'])
            
            x = pd.DataFrame(x, columns=genes, index=samples)
            response = pd.DataFrame(response, columns=['response'], index=samples)
            
            x = x.loc[selected_samples, :]
            response = response.loc[selected_samples, :]
            samples = x.index
            genes = x.columns
            response = response['response'].values
            x = x.values
            
        
        self.x = x
        self.samples = samples
        self.response = response
        self.genes = genes
        
    def get_data(self):
        return self.x, self.samples, self.response, self.genes
    
    def get_train_val_test(self):
        SPLITS_PATH = os.path.join(PROSTATE_DATA_PATH, 'splits')
        
        training_file = 'training_set_{}.csv'.format(self.training_split)
        training_set = pd.read_csv(os.path.join(SPLITS_PATH, training_file))
        
        validation_set = pd.read_csv(os.path.join(SPLITS_PATH, 'validation_set.csv'))
        test_set = pd.read_csv(os.path.join(SPLITS_PATH, 'test_set.csv'))
        
        train_samples = list(set(self.samples).intersection(training_set.id))
        val_samples = list(set(self.samples).intersection(validation_set.id))
        test_samples = list(set(self.samples).intersection(test_set.id))
        
        train_indices = self.samples.isin(train_samples)
        val_indices = self.samples.isin(val_samples)
        test_indices = self.samples.isin(test_samples)
        
        X_train = self.x[train_indices]
        X_test = self.x[test_indices]
        X_val = self.x[val_indices]
        
        y_train = self.response[train_indices]
        y_test = self.response[test_indices]
        y_val = self.response[val_indices]
        
        train_samples = self.samples[train_indices]
        test_samples =  self.samples[test_indices]
        val_samples = self.samples[val_indices]
        
        return X_train, X_val, X_test, y_train, y_val, y_test, train_samples.copy(), val_samples, test_samples.copy(), self.genes