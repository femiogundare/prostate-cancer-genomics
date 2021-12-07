# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 18:51:58 2021

@author: femiogundare
"""

import logging
from sklearn import metrics
import tensorflow as tf
from tensorflow.keras import backend as K


def evaluate(y_test, y_pred, y_pred_score=None):
    accuracy = metrics.accuracy_score(y_test, y_pred)
    
    if y_pred_score is None:
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
    else:
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_score, pos_label=1)
        
    auc = metrics.auc(fpr, tpr)
    f1 = metrics.f1_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    logging.info(metrics.classification_report(y_test, y_pred))

    aupr = metrics.average_precision_score(y_test, y_pred_score)
    logging.info(
        '--accuracy: {0:.2f} precision: {1:.2f} auc: {2:.2f} f1: {3:.2f} aupr {4:.2f} recall: {5:.2f}'.format(accuracy, precision, auc,
                                                                                              f1, aupr, recall))
    
    score = {}
    score['accuracy'] = accuracy
    score['precision'] = precision
    score['auc'] = auc
    score['f1'] = f1
    score['aupr'] = aupr
    score['recall'] = recall
    
    return score



def evaluate_regression(y_true, y_pred, **kwargs):
    var = metrics.explained_variance_score(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    median_absolute_error = metrics.median_absolute_error(y_true, y_pred)
    mean_squared_log_error = metrics.mean_squared_log_error(y_true, y_pred)
    mean_squared_error = metrics.mean_squared_error(y_true, y_pred)
    mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred)
    score = {}
    score['explained variance'] = var
    score['precision'] = r2
    score['median_absolute_error'] = median_absolute_error
    score['mean_squared_log_error'] = mean_squared_log_error
    score['mean_squared_error'] = mean_squared_error
    score['mean_absolute_error'] = mean_absolute_error
    
    return score



# Custom R2-score metrics for keras backend
def r2_keras(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))



def correlation_coefficient(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x - mx, y - my
    r_num = K.sum(tf.multiply(xm, ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return 1 - K.square(r)