# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 14:55:56 2021

@author: femiogundare
"""

import logging


def construct_model(model_params_dict):
    model_type = model_params_dict['type']
    p = model_params_dict['params']
    
    if model_type == 'svr':
        from sklearn.svm import SVR
        model = SVR(max_iter=5000, **p)
    
    if model_type == 'knn':
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(**p)
    
    if model_type == 'svc':
        from sklearn.svm import SVC
        model = SVC(max_iter=5000, **p)
        
    if model_type == 'linear_svc':
        from sklearn.svm import LinearSVC
        model = LinearSVC(max_iter=5000, **p)
        
    if model_type == 'multinomial':
        from sklearn.naive_bayes import MultinomialNB
        model = MultinomialNB(**p)
        
    if model_type == 'nearest_centroid':
        from sklearn.neighbors import NearestCentroid
        model = NearestCentroid(**p)
        
    if model_type == 'bernoulli':
        from sklearn.naive_bayes import BernoulliNB
        model = BernoulliNB(**p)
        
    if model_type == 'sgd':
        from sklearn.linear_model import SGDClassifier
        model = SGDClassifier(**p)
        
    if model_type == 'gaussian_process':
        from sklearn.gaussian_process import GaussianProcessClassifier
        model = GaussianProcessClassifier(**p)
        
    if model_type == 'decision_tree':
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(**p)
        
    if model_type == 'random_forest':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(**p)

    if model_type == 'adaboost':
        from sklearn.ensemble import AdaBoostClassifier
        model = AdaBoostClassifier(**p)
        
        
    if model_type == 'ridge_classifier':
        from sklearn.linear_model import RidgeClassifier
        model = RidgeClassifier(**p)

    elif model_type == 'ridge':
        from sklearn.linear_model import Ridge
        model = Ridge(**p)


    elif model_type == 'elastic':
        from sklearn.linear_model import ElasticNet
        model = ElasticNet(**p)
        
    elif model_type == 'lasso':
        from sklearn.linear_model import Lasso
        model = Lasso(**p)
        
    elif model_type == 'extratrees':
        from sklearn.ensemble import ExtraTreesClassifier
        model = ExtraTreesClassifier(**p)

    elif model_type == 'randomizedLR':
        from sklearn.linear_model import RandomizedLogisticRegression
        model = RandomizedLogisticRegression(**p)
        
    elif model_type == 'RandomForestRegressor':
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(**p)
    
    elif model_type == 'logistic':
        logging.info('model class {}'.format(model_type))
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()

    elif model_type == 'nn':
        from model import neural_network as nn
        model = nn.Model(**p)

    return model



def get_model(params):
    if type(params['params']) == dict:
        model = construct_model(params)
    else:
        model = params['params']
    return model