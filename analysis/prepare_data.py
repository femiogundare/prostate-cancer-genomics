# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 22:36:57 2021

@author: femiogundare
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import PROSTATE_LOG_PATH
from analysis.data_extraction_utils import get_node_importance, get_link_weights_df, get_data, get_degrees, adjust_coef_with_graph_degree
from utilities.loading_utils import DataModelLoader

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

current_dir = os.path.dirname(os.path.realpath(__file__))
saving_dir = os.path.join(current_dir, '_extracted')

if not os.path.exists(saving_dir):
    os.makedirs(saving_dir)

    

def save_gradient_importance(node_weights_dfs, node_weights_samples_dfs, layers, samples):
    for i, layer in enumerate(layers[:-1]):
        layer_node_weight = node_weights_dfs[layer]
        filename = os.path.join(saving_dir, 'gradient_importance_{}.csv'.format(i))
        layer_node_weight.to_csv(filename)
        
    for i, layer in enumerate(layers[:-1]):
        layer_node_weight_samples = node_weights_samples_dfs[layer]
        if i > 0:
            layer_node_weight_samples['ind'] = samples
            layer_node_weight_samples = layer_node_weight_samples.set_index('ind')
            filename = os.path.join(saving_dir, 'gradient_importance_detailed_{}.csv'.format(i))
            layer_node_weight_samples.to_csv(filename)
            
            
def save_link_weights(link_weights_df, layers):
    for i, layer in enumerate(layers):
        link = link_weights_df[layer]
        filename = os.path.join(saving_dir, 'link_weights_{}.csv'.format(i))
        link.to_csv(filename)
        
        
def save_activation(layers_outputs_dict, feature_names, samples):
    for layer_name, layer_output in sorted(layers_outputs_dict.items()):
        if layer_name.startswith('h'):
            print(layer_name, layer_output.shape)
            l = int(layer_name[1:])
            features = feature_names[layer_name]
            layer_output_df = pd.DataFrame(layer_output, index=samples, columns=features)
            layer_output_df = layer_output_df.round(decimals=3)
            filename = os.path.join(saving_dir, 'activation_{}.csv'.format(l + 1))
            layer_output_df.to_csv(filename)
            
            
def save_graph_stats(degrees, fan_outs, fan_ins, layers):
    i = 1

    df = pd.concat([degrees[0], fan_outs[0]], axis=1)
    df.columns = ['degree', 'fan_out']
    df['fan_in'] = 0
    filename = os.path.join(saving_dir, 'graph_stats_{}.csv'.format(i))
    df.to_csv(filename)

    for i, (d, fin, fout) in enumerate(zip(degrees[1:], fan_ins, fan_outs[1:])):
        df = pd.concat([d, fin, fout], axis=1)
        df.columns = ['degree', 'fan_in', 'fan_out']
        filename = os.path.join(saving_dir, 'graph_stats_{}.csv'.format(i + 2))
        df.to_csv(filename)
        
        
     
def extract(nn_model, X, y, samples, feature_names, importance_type, layers, target):
    response = pd.DataFrame(y, index=samples, columns=['response'])
    filename = os.path.join(saving_dir, 'response.csv')
    response.to_csv(filename)
    print('Saving gradient importance...')
    node_weights, node_weights_samples_dfs = get_node_importance(nn_model, X, y, importance_type[0], target)
    save_gradient_importance(node_weights, node_weights_samples_dfs, layers, samples)

    print('Saving link weights...')
    link_weights_df = get_link_weights_df(nn_model.model, feature_names, layers)
    save_link_weights(link_weights_df, layers[1:])

    print('Saving activation...')
    layers_outputs_dict = nn_model.get_layer_outputs(X)
    save_activation(layers_outputs_dict, feature_names, samples)

    print('Saving graph stats...')
    stats = get_degrees(link_weights_df, layers[1:])
    #keys = np.sort(stats.keys())
    keys = sorted(stats.keys())
    for k in keys:
        filename = os.path.join(saving_dir, 'graph_stats_{}.csv'.format(k))
        stats[k].to_csv(filename)
        
    print('Adjusting weights with graph stats...')
    degrees = []
    for k in keys:
        degrees.append(stats[k].degree.to_frame(name='coef_graph'))
    
    print(node_weights)
    print(stats)
    print(layers[1:-1])
    
    
    adjusted_node_importances = adjust_coef_with_graph_degree(node_weights, stats, layers[1:-1], saving_dir)
    #print(adjusted_node_importances)
    filename = os.path.join(saving_dir, 'node_importance_graph_adjusted.csv')
    adjusted_node_importances.to_csv(filename)     
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
"""    
base_dir = os.path.join(PROSTATE_LOG_PATH, 'prostate_net')
model_name = 'onesplit_average_reg_10_tanh_test'
importance_type = ['deepexplain_deeplift']
target = 'o6'
use_data = 'Train'
dropAR = False
layers = ['inputs', 'h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'o_linear6']


def prepare():
    
    from sklearn.metrics import roc_auc_score, accuracy_score
    
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import random
    import tensorflow as tf
    print('***************************************************************************************Tensorflow version', tf.__version__)
    random_seed = 234
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    tf.compat.v1.set_random_seed(random_seed)
    session_config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    session = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_config)
    tf.compat.v1.keras.backend.set_session(session)
    
    
    print('Getting data and saving response...')
    model_dir = os.path.join(base_dir, model_name)
    model_file = 'P-net_ALL'
    params_file = os.path.join(model_dir, model_file + '_params.yml')
    loader = DataModelLoader(params_file)
    nn_model = loader.get_model(model_file)
    feature_names = nn_model.feature_names
    X, y, samples = get_data(loader, use_data, dropAR)
    predict = nn_model.predict(X)
    
    if (type(predict) == list):
        if len(predict) > 1:
            predict = np.mean(np.array(predict), axis=0)
        else:
            predict = predict[-1]


    
    
    
    
    
    
    print('X unique', np.unique(X))
    print('y unique', np.unique(y))
    print('Model X unique', np.unique(nn_model.predict(X)))
    print('AUC Score: {}'.format(roc_auc_score(y, predict)))
    print('Accuracy Score: {}'.format(accuracy_score(y, predict)))
    
    response = pd.DataFrame(y, index=samples, columns=['response'])
    filename = os.path.join(saving_dir, 'response.csv')
    response.to_csv(filename)
    
    print('Saving gradient importance...')
    node_weights, node_weights_samples_dfs = get_node_importance(nn_model, X, y, importance_type[0], target)
    save_gradient_importance(node_weights, node_weights_samples_dfs, layers, samples)

    print('Saving link weights...')
    link_weights_df = get_link_weights_df(nn_model.model, feature_names, layers)
    save_link_weights(link_weights_df, layers[1:])

    print('Saving activation...')
    layers_outputs_dict = nn_model.get_layer_outputs(X)
    save_activation(layers_outputs_dict, feature_names, samples)

    print('Saving graph stats...')
    stats = get_degrees(link_weights_df, layers[1:])
    #keys = np.sort(stats.keys())
    keys = sorted(stats.keys())
    for k in keys:
        filename = os.path.join(saving_dir, 'graph_stats_{}.csv'.format(k))
        stats[k].to_csv(filename)
        
    print('Adjusting weights with graph stats...')
    degrees = []
    for k in keys:
        degrees.append(stats[k].degree.to_frame(name='coef_graph'))
    
    print(node_weights)
    print(stats)
    print(layers[1:-1])
    
    
    adjusted_node_importances = adjust_coef_with_graph_degree(node_weights, stats, layers[1:-1], saving_dir)
    #print(adjusted_node_importances)
    filename = os.path.join(saving_dir, 'node_importance_graph_adjusted.csv')
    adjusted_node_importances.to_csv(filename)
    
    
    
if __name__ == "__main__":
    prepare()
"""