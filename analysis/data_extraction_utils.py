# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 01:32:15 2021

@author: femiogundare
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import REACTOME_PATHWAY_PATH
from model.model_utils import get_coef_importance
from model.custom_layers import SparseTF


def get_data(data_loader, use_data, dropAR):
    X_train, X_test, y_train, y_test, samples_train, samples_test, genes = data_loader.get_data()
    
    if use_data == 'All':
        X = np.concatenate([X_train, X_test], axis=0)
        y = np.concatenate([y_train, y_test], axis=0)
        samples = list(samples_train) + list(samples_test)
        
    elif use_data == 'Test':
        X = X_test
        y = y_test
        samples = list(samples_test)
        
    elif use_data == 'Train':
        X = X_train
        y = y_train
        samples = list(samples_train)

    if dropAR:
        x = pd.DataFrame(X, columns=genes)
        data_types = x.columns.levels[1].unique()
        print(data_types)
        if 'cnv' in data_types:
            ind = (x[('AR', 'cnv')] <= 0.) & (x[('AR', 'important_mutations')] == 0)
        elif 'cnv_amplification' in data_types:
            ind = (x[('AR', 'cnv_amplification')] <= 0.) & (x[('AR', 'important_mutations')] == 0)

        if len(ind.shape) > 1:
            ind = ind.all(axis=1)
        
        x = x.iloc[ind.values,]

        X = x.values
        y = y[ind.values]
        samples = [samples[i] for i in ind.values if i]
        
        use_data = use_data + '_dropAR'
        
    return X, y, samples


def get_reactome_pathway_names():
    """
    Returns a dataframe containing reactome ids, names of reactome pathways, and specie (homo sapiens).
    """
    reactome_pathways_df = pd.read_csv(os.path.join(REACTOME_PATHWAY_PATH, 'ReactomePathways.txt'), sep='	', header=None)
    reactome_pathways_df.columns = ['reactome_id', 'pathway_name', 'specie']
    reactome_pathways_df_human = reactome_pathways_df[reactome_pathways_df['specie'] == 'Homo sapiens']
    reactome_pathways_df_human.reset_index(inplace=True)
    return reactome_pathways_df_human


def get_pathway_names(all_nodes):
    pathways_ids_and_names = get_reactome_pathway_names()
    ids = list(pathways_ids_and_names['reactome_id'])
    names = list(pathways_ids_and_names['pathway_name'])
    
    pathways_names = []
    for node in all_nodes:
        if node in ids:
            ind = ids.index(node)
            node_pathway_name = names[ind]
            pathways_names.append(node_pathway_name)
        else:
            pathways_names.append(node)

    return pathways_names

"""
def get_node_importance(nn_model, x_train, y_train, importance_type, target):
    model = nn_model.model
    coefs = get_coef_importance(model, x_train, y_train, target=target, feature_importance=importance_type, detailed=True)
    print(type(coefs))
    if type(coefs) is tuple:
        coef, coef_detailed = coefs
    else:
        coef = coefs
        coef_detailed = [c.T for c in coef]
        
    node_weights_dfs = {}
    node_weights_samples_dfs = {}
    
    for i, k in enumerate(nn_model.feature_names.keys()):
        nodes = nn_model.feature_names[k]
        w = coef[k]
        w_samples = coef_detailed[k]
        features = get_pathway_names(nodes)
        df = pd.DataFrame(abs(w.ravel()), index=nodes, columns=['coef'])
        layer = pd.DataFrame(index=nodes)
        layer['layer'] = i
        node_weights_dfs[k] = df
        df_samples = pd.DataFrame(w_samples, columns=features)
        node_weights_samples_dfs[k] = (df_samples)
    
    return node_weights_dfs, node_weights_samples_dfs
"""


def get_node_importance(nn_model, x_train, y_train, importance_type, target):
    
    model = nn_model.model
    coefs = get_coef_importance(model, x_train, y_train, target=target, feature_importance=importance_type, detailed=True)
    
    #coefs = get_coef_importance(nn_model, x_train, y_train, target=target, feature_importance=importance_type, detailed=True)
    print(type(coefs))
    if type(coefs) is tuple:
        coef, coef_detailed = coefs
    else:
        coef = coefs
        coef_detailed = [c.T for c in coef]
        
    node_weights_dfs = {}
    node_weights_samples_dfs = {}
    
    for i, k in enumerate(nn_model.feature_names.keys()):
        nodes = nn_model.feature_names[k]
        w = coef[k]
        w_samples = coef_detailed[k]
        features = get_pathway_names(nodes)
        """
        if k == 'inputs':
            data_types = nn_model.data_types
            #print(len(data_types))
            index=[nodes, data_types]
        else:
            index=nodes
        """
        df = pd.DataFrame(abs(w.ravel()), index=nodes, columns=['coef'])
        layer = pd.DataFrame(index=nodes)
        layer['layer'] = i
        node_weights_dfs[k] = df
        df_samples = pd.DataFrame(w_samples, columns=features)
        node_weights_samples_dfs[k] = (df_samples)
    
    return node_weights_dfs, node_weights_samples_dfs


def get_layer_weights(layer):
    print(layer.get_weights())
    w = layer.get_weights()[0]
    if type(layer) == SparseTF:
        row_ind = layer.nonzero_ind[:, 0]
        col_ind = layer.nonzero_ind[:, 1]
        w = csr_matrix((w, (row_ind, col_ind)), shape=layer.kernel_shape)
        w = w.todense()
    return w


def get_link_weights_df(model, features, layer_names):
    link_weights_df = {}
    
    for i, layer_name in enumerate(layer_names[1:]):
        layer = model.get_layer(layer_name)
        w = get_layer_weights(layer)
        layer_index = layer_names.index(layer_name)
        previous_layer_name = layer_names[layer_index - 1]
        print('Previous layer name: {}'.format(previous_layer_name))
        print('Current layer name: {}'.format(layer_name))
        
        if (i == 0) or (i == (len(layer_names) - 2)):
            cols = ['root']
        else:
            cols = features[layer_name]
            
        rows = features[previous_layer_name]
        w_df = pd.DataFrame(w, index=rows, columns=cols)
        link_weights_df[layer_name] = w_df
        
    return link_weights_df


"""
def get_link_weights(model):
    layers = model.layers
    n = len(layers)
    hidden_layers_weights = []
    next = 0
    for i, l in enumerate(layers):
        if l.name.startswith('h') or l.name == 'o_linear6':
            w = l.get_weights()[0]
            if type(l) == SparseTF:
                row_ind = l.nonzero_ind[:, 0]
                col_ind = l.nonzero_ind[:, 1]
                w = csr_matrix((w, (row_ind, col_ind)), shape=l.kernel_shape)
                w = w.todense()
            hidden_layers_weights.append(w)
            print(l.name, len(l.get_weights()), w.shape)
            
    return hidden_layers_weights
"""


def get_degrees(maps, layers):
    stats = {}
    
    for i, (l1, l2) in enumerate(zip(layers[1:], layers[2:])):
        layer_1 = maps[l1]
        layer_2 = maps[l2]
        
        layer_1[layer_1 != 0] = 1.
        layer_2[layer_2 != 0] = 1.
        
        fan_out_1 = layer_1.abs().sum(axis=1)
        fan_in_1 = layer_1.abs().sum(axis=0)

        fan_out_2 = layer_2.abs().sum(axis=1)
        fan_in_2 = layer_2.abs().sum(axis=0)
        
        if i == 0:
            layer = layers[0]
            df = pd.concat([fan_out_1, fan_out_1], keys=['degree', 'fan_out'], axis=1)
            df['fan_in'] = 1.
            stats[layer] = df
            
        print('{}- layer {} : fan-in {}, fan-out {}'.format(i, l1, fan_in_1.shape, fan_out_2.shape))
        print('{}- layer {} : fan-in {}, fan-out {}'.format(i, l1, fan_in_2.shape, fan_out_1.shape))
        
        df = pd.concat([fan_in_1, fan_out_2], keys=['fan_in', 'fan_out'], axis=1)
        df['degree'] = df['fan_in'] + df['fan_out']
        stats[l1] = df

    return stats

"""
def adjust_coef_with_graph_degree(node_importance_dfs, stats, layer_names, saving_dir):
    adjusted_node_importances = []
    for i, layer in enumerate(layer_names):
        node_importance = node_importance_dfs[layer]
        #print(node_importance.columns)
        #print(node_importance.head())
        #print('First....................................................................................')
        #print(node_importance.index)
        degree = stats[layer]['degree'].to_frame(name='degree')
        #print(degree.columns)
        #print(degree.head())
        #print('\n\n', degree.index)
        
        node_importance.index = get_pathway_names(node_importance.index)
        #print(node_importance.columns)
        #print(node_importance.head())
        #print('\n\n\nSecond..............................................................................')
        #print(node_importance.index)
        degree.index = get_pathway_names(degree.index)
        #print(degree.columns)
        #print(degree.head())
        #print('\n\n', degree.index)
        df = node_importance.join(degree, how='inner')
        #print('\n\n', df)
        
        #print('Layer {}*********************************************************************************'.format(layer))
        #print('1......................')
        #print(df)
        
        mean = df.degree.mean()
        std = df.degree.std()
        indices = df.degree > mean + 5*std
        df_degree = df.degree.copy()
        df_degree[~indices] = df_degree[~indices] = 1.0
        
        df['adjusted_coef'] = df['coef'] / df_degree
        z = (df['adjusted_coef'] - df['adjusted_coef'].mean()) / df['adjusted_coef'].std(ddof=0)
        df['adjusted_coef_z_score'] = z
        
        #print('\n2..........................')
        #print(df)
        
        
        # Graph coefficient -------------------------- DEGREE
        z1 = (df['degree'] - df['degree'].mean()) / df['degree'].std(ddof=0)
        # Gradient coefficient ------------------------- COEF (from model)
        z2 = (df['coef'] - df['coef'].mean()) / df['coef'].std(ddof=0)
        
        coef_degree_diff = z2 -z1
        df['coef_combined'] = (coef_degree_diff - coef_degree_diff.mean()) / coef_degree_diff.std(ddof=0)
        
        print('\n3...............................')
        print(df)
        
        filename = os.path.join(saving_dir, 'layer_{}_graph_adjusted.csv'.format(i))
        df.to_csv(filename)
        
        df['layer'] = i + 1
        adjusted_node_importances.append(df)
        
        print('4................................')
        print(df)

        
    adjusted_node_importances = pd.concat(adjusted_node_importances)
    adjusted_node_importances = adjusted_node_importances.groupby(adjusted_node_importances.index).min()
    
    return adjusted_node_importances
"""

def adjust_layer(df):
    # graph coef
    z1 = df.coef_graph
    z1 = (z1 - z1.mean()) / z1.std(ddof=0)

    # gradient coef
    z2 = df.coef
    z2 = (z2 - z2.mean()) / z2.std(ddof=0)

    z = z2 - z1

    z = (z - z.mean()) / z.std(ddof=0)
    x = np.arange(len(z))
    df['coef_combined2'] = z
    return df

def adjust_coef_with_graph_degree(node_importance_dfs, stats, layer_names, saving_dir):
    ret = []
    # for i, (grad, graph) in enumerate(zip(node_importance_dfs, degrees)):
    for i, l in enumerate(layer_names):
        #print l
        grad = node_importance_dfs[l]
        graph = stats[l]['degree'].to_frame(name='coef_graph')

        graph.index = get_pathway_names(graph.index)
        grad.index = get_pathway_names(grad.index)
        d = grad.join(graph, how='inner')

        mean = d.coef_graph.mean()
        std = d.coef_graph.std()
        ind = d.coef_graph > (mean + 5 * std)
        divide = d.coef_graph.copy()
        divide[~ind] = divide[~ind] = 1.
        d['coef_combined'] = d.coef / divide
        z = d.coef_combined
        z = (z - z.mean()) / z.std(ddof=0)
        d['coef_combined_zscore'] = z
        d = adjust_layer(d)
        #         d['coef_combined'] = d['coef_combined']/sum(d['coef_combined'])
        filename = os.path.join(saving_dir, 'layer_{}_graph_adjusted.csv'.format(i))
        d.to_csv(filename)
        d['layer'] = i + 1
        ret.append(d)
    node_importance = pd.concat(ret)
    node_importance = node_importance.groupby(node_importance.index).min()
    return node_importance