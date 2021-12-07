# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 09:39:01 2021

@author: femiogundare
"""

import os
import sys
import logging
import time
import pickle
import numpy as np

from tensorflow.keras.models import Sequential
from matplotlib import pyplot as plt
from pathlib import Path


def save_model(model, filename):
    print('Saving model in', filename)
    file = filename + '.pkl'
    f = open(file, 'wb')
    sys.setrecursionlimit(100000)
    pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()
    
    
def load_model(filename):
    print('Loading model from', filename)
    file = filename + '.pkl'
    start_time = time.time()
    model = pickle.load(filename)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Time taken to load model: {:.4f}'.format(elapsed_time))
    return model


def print_model(model, level=1):
    for i, layer in enumerate(model.layers):
        indent = '  ' * level + '-'
        if type(layer) == Sequential:
            logging.info('{} {} {} {}'.format(indent, i, layer.name, layer.output_shape))
            print_model(layer, level + 1)
        else:
            logging.info('{} {} {} {}'.format(indent, i, layer.name, layer.output_shape))
            
            
def get_layers(model, level=1):
    layers = []
    for i, layer in enumerate(model.layers):
        if type(layer) == Sequential:
            layers.extend(get_layers(layer, level + 1))
        else:
            layers.append(layer)

    return layers



from model.coef_weights_utils import get_gradient_layer, get_shap_scores_layer, get_shap_scores, get_deep_explain_scores, \
    get_deep_explain_score_layer, get_gradient_weights, get_gradient_weights_with_repeated_output, \
        get_weights_linear_model, get_gradient, get_weights_gradient_outcome, get_permutation_weights, get_deconstruction_weights
            
            

def get_coef_importance(model, X_train, y_train, target, feature_importance, detailed=True, **kwargs):
    """
    if feature_importance.startswith('skf'):
        coef_ = get_skf_weights(model, X_train, y_train, feature_importance)
    """
    if feature_importance == 'loss_gradient':
        coef_ = get_gradient_weights(model, X_train, y_train, signed=False, detailed=detailed,
                                     normalize=True)  # use total loss
    elif feature_importance == 'loss_gradient_signed':
        coef_ = get_gradient_weights(model, X_train, y_train, signed=True, detailed=detailed,
                                     normalize=True)  # use total loss
    elif feature_importance == 'gradient_outcome':
        coef_ = get_weights_gradient_outcome(model, X_train, y_train, target, multiply_by_input=False, signed=False)
    elif feature_importance == 'gradient_outcome_signed':
        coef_ = get_weights_gradient_outcome(model, X_train, y_train, target=target, detailed=detailed,
                                             multiply_by_input=False, signed=True)
    elif feature_importance == 'gradient_outcome*input':
        coef_ = get_weights_gradient_outcome(model, X_train, y_train, target, multiply_by_input=True, signed=False)
    elif feature_importance == 'gradient_outcome*input_signed':
        coef_ = get_weights_gradient_outcome(model, X_train, y_train, target, multiply_by_input=True, signed=True)

    elif feature_importance.startswith('deepexplain'):
        method = feature_importance.split('_')[1]
        coef_ = get_deep_explain_scores(model, X_train, y_train, target, method_name=method, detailed=detailed,
                                        **kwargs)

    elif feature_importance.startswith('shap'):
        method = feature_importance.split('_')[1]
        coef_ = get_shap_scores(model, X_train, y_train, target, method_name=method, detailed=detailed)


    elif feature_importance == 'gradient_with_repeated_outputs':
        coef_ = get_gradient_weights_with_repeated_output(model, X_train, y_train, target)
    elif feature_importance == 'permutation':
        coef_ = get_permutation_weights(model, X_train, y_train)
    elif feature_importance == 'linear':
        coef_ = get_weights_linear_model(model, X_train, y_train)
    elif feature_importance == 'one_to_one':
        weights = model.layers[1].get_weights()
        switch_layer_weights = weights[0]
        coef_ = np.abs(switch_layer_weights)
    else:
        coef_ = None
    return coef_


def apply_models(models, inputs):
    output = inputs
    for m in models:
        output = m(output)

    return output


def plot_channels(history, channels, filename, folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    plt.figure()
    for k in channels:
        v = history[k]
        plt.plot(v)
    plt.legend(channels)
    filename = os.path.join(folder_name, filename)
    plt.savefig(filename)
    plt.close()
    
    
    
def plot_history(history, folder_name):
    keys = history.keys()

    losses = [x for x in keys if ('_loss' in x) and (x != 'val_loss')]
    val_losses = [x for x in losses if 'val_' in x]
    train_losses = [x for x in losses if ('val_' not in x) and (x != 'loss')]
    # train_losses = [x.replace('val_', '') for x in val_losses ]

    monitors = [x for x in keys if 'loss' not in x]
    val_monitors = [x for x in monitors if 'val_' in x]
    train_monitors = [x for x in monitors if ('val_' not in x) and (x != 'loss') and (x != 'lr')]
    # train_monitors= [x.replace('val_', '') for x in val_monitors]

    monitors.sort()
    val_monitors.sort()
    train_monitors.sort()

    train_losses.sort()
    val_losses.sort()

    #print(val_losses)
    #print(train_losses)
    #print(monitors)

    plot_channels(history, val_monitors, 'val_monitors', folder_name)
    plot_channels(history, train_monitors, 'train_monitors', folder_name)
    # plot_channels(history, ['val_loss', 'loss'], 'loss')
    for v, t in zip(val_monitors, train_monitors):
        plot_channels(history, [v, t], t, folder_name)

    plot_channels(history, val_losses, 'validation_loss', folder_name)
    plot_channels(history, train_losses, 'training_loss', folder_name)

    if 'val_loss' in keys:
        plot_channels(history, ['val_loss', 'loss'], 'loss', folder_name)
    else:
        plot_channels(history, ['loss'], 'loss', folder_name)

    for v, t in zip(val_losses, train_losses):
        plot_channels(history, [v, t], t, folder_name)
    pass