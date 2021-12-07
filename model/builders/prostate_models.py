# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 17:11:03 2021

@author: femiogundare
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from tensorflow import keras

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from data_scripts.data_access import Data
from data_scripts.pathways.gmt_pathway import get_KEGG_map
from model.builders.builders_utils import get_pnet
from model.custom_layers import f1_score, Diagonal, SparseTF
from model.model_utils import print_model, get_layers


# Assumes the first node is connected to the first n nodes and so on
def build_pnet(optimizer, w_reg, w_reg_outcomes, add_unk_genes=True, sparse=True, loss_weights=1.0, dropout=0.5,
                use_bias=False, activation='tanh', loss='binary_crossentropy', data_params=None, n_hidden_layers=1,
                direction='root_to_leaf', batch_normal=False, kernel_initializer='glorot_uniform', shuffle_genes=False,
                attention=False, dropout_testing=False, non_neg=False, repeated_outcomes=True, sparse_first_layer=True):
    data = Data(**data_params)
    x, samples, response, genes = data.get_data()
    print('Shape of the data matrix: {}'.format(x.shape))
    print('Number of samples: {}'.format(samples.shape[0]))
    print('Number of responses: {}'.format(response.shape[0]))
    print('Number of genes: {}'.format(genes.shape[0]))
    
    features = genes
    
    if loss == 'binary_crossentropy':
        activation_decision = 'sigmoid'
    else:
        activation_decision = 'linear'
        
    print('Activation decision: {}'.format(activation_decision))
        
    logging.info('x shape {} , samples shape {} response shape {} genes shape {}'.format(x.shape, samples.shape, response.shape, genes.shape))

    n_features = x.shape[1]
    
    if hasattr(genes, 'levels'):
        genes = genes.levels[0]
    else:
        genes = genes
        
    ins = keras.layers.Input(shape=(n_features,), dtype='float32', name='inputs')
    
    outcome, decision_outcomes, feature_n = get_pnet(ins,
                                                     features=features,
                                                     genes=genes,
                                                     n_hidden_layers=n_hidden_layers,
                                                     direction=direction,
                                                     activation=activation,
                                                     activation_decision=activation_decision,
                                                     w_reg=w_reg,
                                                     w_reg_outcomes=w_reg_outcomes,
                                                     dropout=dropout,
                                                     sparse=sparse,
                                                     add_unk_genes=add_unk_genes,
                                                     batch_normal=batch_normal,
                                                     sparse_first_layer=sparse_first_layer,
                                                     use_bias=use_bias,
                                                     kernel_initializer=kernel_initializer,
                                                     shuffle_genes=shuffle_genes,
                                                     attention=attention,
                                                     dropout_testing=dropout_testing,
                                                     non_neg=non_neg

                                                     )
    
    feature_names = feature_n
    feature_names['inputs'] = features

    print('Compiling...')
    if repeated_outcomes:
        outcome = decision_outcomes
    else:
        outcome = decision_outcomes[-1]

    model = keras.models.Model(inputs=[ins], outputs=outcome)

    if type(outcome) == list:
        n_outputs = len(outcome)
    else:
        n_outputs = 1

    if type(loss_weights) == list:
        loss_weights = loss_weights
    else:
        loss_weights = [loss_weights] * n_outputs

    print('Loss_weights', loss_weights)
    model.compile(optimizer=optimizer,
                  loss=['binary_crossentropy'] * n_outputs, metrics=[f1_score], loss_weights=loss_weights)

    logging.info('Done compiling...')

    print_model(model)
    logging.info(model.summary())
    logging.info('Number of trainable parameters of the model is {}'.format(model.count_params()))
    return model, feature_names
    

def apply_models(models, inputs):
    output = inputs
    for m in models:
        output = m(output)

    return output



def get_clinical_network(ins, n_features, n_hids, activation):
    layers = []
    for i, n in enumerate(n_hids):
        if i == 0:
            layer = keras.layers.Dense(n, input_shape=(n_features,), activation=activation, 
                                       kernel_regularizer=keras.regularizers.l2(0.001),
                                       name='h_clinical' + str(i)
                                       )
        else:
            layer = keras.layers.Dense(n, activation=activation, kernel_regularizer=keras.regularizers.l2(0.001), 
                                       name='h_clinical' + str(i)
                                       )

        layers.append(layer)
        drop = 0.5
        layers.append(keras.layers.Dropout(drop, name='droput_clinical_{}'.format(i)))

    merged = apply_models(layers, ins)
    output_layer = keras.layers.Dense(1, activation='sigmoid', name='clinical_out')
    outs = output_layer(merged)

    return outs



def build_pnet_account_for(optimizer, w_reg, w_reg_outcomes, add_unk_genes=True, sparse=True, loss_weights=1.0,
                            dropout=0.5,
                            use_bias=False, activation='tanh', loss='binary_crossentropy', data_params=None,
                            n_hidden_layers=1,
                            direction='root_to_leaf', batch_normal=False, kernel_initializer='glorot_uniform',
                            shuffle_genes=False,
                            attention=False, dropout_testing=False, non_neg=False, repeated_outcomes=True,
                            sparse_first_layer=True):

    data = Data(**data_params)
    x, samples, response, genes = data.get_data()
    assert len(
        genes.levels) == 3, "Expect to have pandas dataframe with 3 levels [{'clinical, 'genomics'}, genes, features] "

    
    x_df = pd.DataFrame(x, columns=genes, index=samples)
    genomics_label = list(x_df.columns.levels[0]).index(u'genomics')
    genomics_ind = x_df.columns.codes[0] == genomics_label
    genomics = x_df['genomics']
    features_genomics = genomics.columns.remove_unused_levels()
    

    if loss == 'binary_crossentropy':
        activation_decision = 'sigmoid'
    else:
        activation_decision = 'linear'
        
    logging.info('x shape {} , samples shape {} response shape {} genes shape {}'.format(x.shape, samples.shape, response.shape, genes.shape))

    n_features = x_df.shape[1]
    n_features_genomics = len(features_genomics)

    if hasattr(features_genomics, 'levels'):
        genes = features_genomics.levels[0]
    else:
        genes = features_genomics

    #print("n_features", n_features)
    #print("n_features_genomics", n_features_genomics)
    #print("Genes", len(genes), genes)

    ins = keras.layers.Input(shape=(n_features,), dtype='float32', name='inputs')

    ins_genomics = keras.layers.Lambda(lambda x: x[:, 0:n_features_genomics])(ins)
    ins_clinical = keras.layers.Lambda(lambda x: x[:, n_features_genomics:n_features])(ins)

    clinical_outs = get_clinical_network(ins_clinical, n_features, n_hids=[50, 1], activation=activation)

    outcome, decision_outcomes, feature_n = get_pnet(ins_genomics,
                                                     features=features_genomics,
                                                     genes=genes,
                                                     n_hidden_layers=n_hidden_layers,
                                                     direction=direction,
                                                     activation=activation,
                                                     activation_decision=activation_decision,
                                                     w_reg=w_reg,
                                                     w_reg_outcomes=w_reg_outcomes,
                                                     dropout=dropout,
                                                     sparse=sparse,
                                                     add_unk_genes=add_unk_genes,
                                                     batch_normal=batch_normal,
                                                     sparse_first_layer=sparse_first_layer,
                                                     use_bias=use_bias,
                                                     kernel_initializer=kernel_initializer,
                                                     shuffle_genes=shuffle_genes,
                                                     attention=attention,
                                                     dropout_testing=dropout_testing,
                                                     non_neg=non_neg

                                                     )

    feature_names = feature_n
    feature_names['inputs'] = x_df.columns

    print('Compiling...')

    if repeated_outcomes:
        outcome = decision_outcomes
    else:
        outcome = decision_outcomes[-1]

    outcome_list = outcome + [clinical_outs]

    combined_outcome = keras.layers.Concatenate(axis=-1, name='combine')(outcome_list)
    output_layer = keras.layers.Dense(1, activation='sigmoid', name='combined_outcome')
    combined_outcome = output_layer(combined_outcome)
    outcome = outcome_list + [combined_outcome]
    model = keras.models.Model(inputs=[ins], outputs=outcome)

    if type(outcome) == list:
        n_outputs = len(outcome)
    else:
        n_outputs = 1

    if type(loss_weights) == list:
        loss_weights = loss_weights
    else:
        loss_weights = [loss_weights] * n_outputs

    print('loss_weights', loss_weights)
    model.compile(optimizer=optimizer,
                  loss=['binary_crossentropy'] * n_outputs, metrics=[f1_score], loss_weights=loss_weights)
    logging.info('Done compiling...')

    print_model(model)
    print(get_layers(model))
    logging.info(model.summary())
    logging.info('Number of trainable params of the model is %s' % model.count_params())
    return model, feature_names



def build_dense(optimizer, n_weights, w_reg, activation='tanh', loss='binary_crossentropy', data_params=None):
    print(data_params)

    data = Data(**data_params)
    x, samples, response, genes = data.get_data()
    
    #print(x.shape)
    #print(samples.shape)
    #print(response.shape)
    #print(genes.shape)
    
    features = genes
    if loss == 'binary_crossentropy':
        activation_decision = 'sigmoid'
    else:
        activation_decision = 'linear'
        
    logging.info('x shape {} , samples shape {} response shape {} genes shape {}'.format(x.shape, samples.shape, response.shape, genes.shape))

    feature_names = []
    feature_names.append(features)

    n_features = x.shape[1]

    ins = keras.layers.Input(shape=(n_features,), dtype='float32', name='inputs')
    n = np.ceil(float(n_weights) / float(n_features))
    print(n)
    
    layer1 = keras.layers.Dense(units=int(n), activation=activation, 
                                kernel_regularizer=keras.regularizers.l2(w_reg), name='h0')
    outcome = layer1(ins)
    outcome = keras.layers.Dense(1, activation=activation_decision, name='output')(outcome)
    model = keras.models.Model(inputs=[ins], outputs=outcome)

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy', metrics=[f1_score])
    logging.info('Done compiling')

    print_model(model)
    print(get_layers(model))
    logging.info(model.summary())
    logging.info('Number of trainable params of the model is %s' % model.count_params())
    return model, feature_names



def build_pnet_KEGG(optimizer, w_reg, dropout=0.5, activation='tanh', use_bias=False,
                    kernel_initializer='glorot_uniform', data_params=None, arch=''):
    data = Data(**data_params)
    x, samples, response, genes = data.get_data()
    
    #print(x.shape)
    #print(samples.shape)
    #print(response.shape)
    #print(genes.shape)

    logging.info('x shape {} , samples shape {} response shape {} genes shape {}'.format(x.shape, samples.shape, response.shape, genes.shape))
    
    feature_names = {}
    feature_names['inputs'] = genes
    # feature_names.append(cols)

    n_features = x.shape[1]
    if hasattr(genes, 'levels'):
        genes = genes.levels[0]
    else:
        genes = genes

    feature_names['h0'] = genes
    # feature_names.append(genes)
    decision_outcomes = []
    n_genes = len(genes)
    genes = list(genes)

    layer1 = Diagonal(n_genes, input_shape=(n_features,), activation=activation, 
                      W_regularizer=keras.regularizers.l2(w_reg),
                      use_bias=use_bias, name='h0', kernel_initializer=kernel_initializer)

    ins = keras.layers.Input(shape=(n_features,), dtype='float32', name='inputs')
    layer1_output = layer1(ins)

    decision0 = keras.layers.Dense(1, activation='sigmoid', name='o0'.format(0))(ins)
    decision_outcomes.append(decision0)

    decision1 = keras.layers.Dense(1, activation='sigmoid', name='o{}'.format(1))(layer1_output)
    decision_outcomes.append(decision1)

    mapp, genes, pathways = get_KEGG_map(genes, arch)

    n_genes, n_pathways = mapp.shape
    logging.info('Number of genes: {}, Number of pathways: {}'.format(n_genes, n_pathways))
    

    hidden_layer = SparseTF(n_pathways, mapp, activation=activation, 
                            W_regularizer=keras.regularizers.l2(w_reg),
                            name='h1', kernel_initializer=kernel_initializer,
                            use_bias=use_bias)


    layer2_output = hidden_layer(layer1_output)
    decision2 = keras.layers.Dense(1, activation='sigmoid', name='o2')(layer2_output)
    decision_outcomes.append(decision2)

    feature_names['h1'] = pathways
    # feature_names.append(pathways)
    print('Compiling...')

    model = keras.models.Model(inputs=[ins], outputs=decision_outcomes)

    model.compile(optimizer=optimizer,
                  loss=['binary_crossentropy'] * 3, metrics=[f1_score])
    
    logging.info('Done compiling...')

    print_model(model)
    print(get_layers(model))
    logging.info(model.summary())
    logging.info('Number of trainable params of the model is %s' % model.count_params())
    
    return model, feature_names