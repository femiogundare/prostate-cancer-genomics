# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 11:29:52 2021

@author: femiogundare
"""

import os
import copy
import logging
import numpy as  np
import pandas as pd
import scipy
import yaml
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from data_scripts.data_access import Data
from analysis.prepare_data import extract
from model.model_factory import get_model
from pipeline.pipeline_utilities import get_model_id, get_coef_from_model, get_balanced
from preprocessing import preprocessor
from utilities.evaluate import evaluate, evaluate_regression
from utilities.plots import generate_plots, plot_roc, plot_prc, save_confusion_matrix
from utilities.random_seed import set_random_seeds


def save_model(model, model_name, directory_name):
    filename = os.path.join(directory_name, 'fs')
    logging.info('Saving model {} coef to dir ({})'.format(model_name, filename))
    if not os.path.exists(filename.strip()):
        os.makedirs(filename)
        
    filename = os.path.join(filename, model_name + '.h5')
    logging.info('fS dir ({})'.format(filename))
    model.save_model(filename)
    
    
def get_model_name(model):
    if 'id' in model:
        model_name = model['id']
    else:
        model_name = model['type']
    return model_name



class OneSplitPipeline:
    """
    Splits the data once into a training and test set and performs a series of operations on them.
    """
    def __init__(self, task, data_params, pre_params, feature_params, model_params, pipeline_params, exp_name):
        """
        Parameters:
            
            task: str or list
                The task to be performed. Possible arguements include 'classification_binary' for binary classification tasks,
                and 'regression' for regression tasks.
                
            data_params: dict
                A dictionary containing data parameters.
                
            pre_params: dict
                A dictionary containing the type of preprocessing to be applied on the data. Possible arguments include
                'standard', 'normalize', 'scale', 'tfidf' and 'None'.
                
            model_params: list
                A list containing the parameters of each of the models to be trained on the dataset.
                
            pipeline_params: dict
                A dictionary containing the type of split to be performed - 'one_split' in this case.
                
            exp_name: str
                Path to store results.
                
        """

        self.task = task
        
        if type(data_params) == list:
            self.data_params = data_params
        else:
            self.data_params = [data_params]
            
        self.pre_params = pre_params
        self.features_params = feature_params
        self.model_params = model_params
        self.pipeline_params = pipeline_params
        self.exp_name = exp_name
        
        print(pipeline_params)
        
        if 'save_train' in pipeline_params['params']:
            self.save_train = pipeline_params['params']['save_train']
        else:
            self.save_train = False
            
        if 'eval_dataset' in pipeline_params['params']:
            self.eval_dataset = pipeline_params['params']['eval_dataset']
        else:
            self.eval_dataset = 'validation'
            
        self.prepare_saving_dir()


    def prepare_saving_dir(self):
        self.directory = self.exp_name
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
            
    
    def save_prediction(self, info, y_pred, y_pred_scores, y_test, model_name, training=False):

        if training:
            file_name = os.path.join(self.directory, model_name + '_training.csv')
        else:
            file_name = os.path.join(self.directory, model_name + '_testing.csv')
            
        logging.info("Saving results : %s" % file_name)
        
        print('info', info)
        info = pd.DataFrame(index=info)
        info['pred'] = y_pred
        info['pred_scores'] = y_pred_scores

        if y_test.dtype.fields is not None:
            fields = y_test.dtype.fields
            for f in fields:
                info['y_{}'.format(f)] = y_test[f]
        else:
            info['y'] = y_test
            
        info.to_csv(file_name)
        
        
    def get_list(self, x, cols):
        x_df = pd.DataFrame(x, columns=cols)
        genes = cols.get_level_values(0).unique()
        genes_list = []
        input_shapes = []
        for gene in genes:
            g_df = x_df.loc[:, gene].as_matrix()
            input_shapes.append(g_df.shape[1])
            genes_list.append(g_df)
            
        return genes_list
    
    
    def get_train_test(self, data):
        x_train, x_test, y_train, y_test, info_train, info_test, columns = data.get_train_test()
        balance_train = False
        balance_test = False
        p = self.pipeline_params['params']
        if 'balance_train' in p:
            balance_train = p['balance_train']
        if 'balance_test' in p:
            balance_test = p['balance_test']

        if balance_train:
            x_train, y_train, info_train = get_balanced(x_train, y_train, info_train)
            
        if balance_test:
            x_test, y_test, info_test = get_balanced(x_test, y_test, info_test)
            
        return x_train, x_test, y_train, y_test, info_train, info_test, columns
    
    
    def run(self):
        test_scores = []
        model_names = []
        model_list = []
        y_pred_test_list = []
        y_pred_test_scores_list = []
        y_test_list = []
        
        fig = plt.figure()
        fig.set_size_inches((10, 6))
        
        for data_params in self.data_params:
            data_id = data_params['id']
            logging.info('Loading data....')
            data = Data(**data_params)

            x_train, x_validate_, x_test_, y_train, y_validate_, y_test_, info_train, info_validate_, info_test_, cols = data.get_train_val_test()

            logging.info('Predicting...')
            if self.eval_dataset == 'validation':
                x_t = x_validate_
                y_t = y_validate_
                info_t = info_validate_
            else:
                x_t = np.concatenate((x_test_, x_validate_))
                y_t = np.concatenate((y_test_, y_validate_))
                info_t = info_test_.append(info_validate_)

            logging.info('x_train {}, y_train {} '.format(x_train.shape, y_train.shape))
            logging.info('x_test {}, y_test {} '.format(x_t.shape, y_t.shape))

            # Pre-processing
            logging.info('Preprocessing....')
            x_train, x_test = self.preprocess(x_train, x_t)
            for m in self.model_params:
                # Get model
                model_params_ = copy.deepcopy(m)
                set_random_seeds(random_seed=20080808)
                model = get_model(model_params_)
                logging.info('Fitting...')
                logging.info(model_params_)
                
                if model_params_['type'] == 'nn' and not self.eval_dataset == 'validation':
                    #init_op = tf.compat.v1.global_variables_initializer()

                    #sess = tf.compat.v1.Session()
                    #session.run(init_op)
                    model = model.fit(x_train, y_train, x_validate_, y_validate_)
                    #model = model.fit(np.array(x_train), y_train, np.array(x_validate_), y_validate_)
                else:
                    #model = model.fit(x_train, y_train)
                    model = model.fit(np.array(x_train), np.array(y_train))
                    
                logging.info('Predicting...')

                model_name = get_model_name(model_params_)
                model_name = model_name + '_' + data_id
                model_params_['id'] = model_name
                logging.info('model id: {}'.format(model_name))
                model_list.append((model, model_params_))
                
                y_pred_test, y_pred_test_scores = self.predict(model, x_test, y_t)
                test_score = self.evaluate(y_t, y_pred_test, y_pred_test_scores)
                logging.info('Model name {} -- Test score {}'.format(model_name, test_score))
                test_scores.append(test_score)
                model_names.append(model_name)
                
                logging.info('Saving results...')
                self.save_score(data_params, model_params_, test_score, model_name)
                self.save_prediction(info_t, y_pred_test, y_pred_test_scores, y_t, model_name)
                
                y_test_list.append(y_t)
                y_pred_test_list.append(y_pred_test)
                y_pred_test_scores_list.append(y_pred_test_scores)

                # Saving coef
                self.save_coef([(model, model_params_)], cols)

                # Saving confusion matrix
                cnf_matrix = confusion_matrix(y_t, y_pred_test)
                save_confusion_matrix(cnf_matrix, self.directory, model_name)

                # saving coefs
                logging.info('Saving coef...')
                if hasattr(model, 'save_model'):
                    logging.info('saving coef')
                    save_model(model, model_name, self.directory)

                if self.save_train:
                    y_pred_train, y_pred_train_scores = self.predict(model, x_train, y_train)
                    train_score = self.evaluate(y_train, y_pred_train, y_pred_train_scores)
                    logging.info('Model {} -- Train score {}'.format(model_name, train_score))
                    self.save_prediction(info_train, y_pred_train, y_pred_train_scores, y_train, model_name,
                                         training=True)
                    
                    
                if self.directory.split('/')[-1] == 'onesplit_average_reg_10_tanh_test' and model_params_['type'] == 'nn':
                    params_file = os.path.join(self.directory, model_name + '_params.yml')
                    feature_names = model.feature_names
                    print('***********Feature names : {}'.format(feature_names))
                    X, y, samples = x_t, y_t,info_t
                    importance_type = ['deepexplain_deeplift']
                    target = 'o6'
                    layers = ['inputs', 'h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'o_linear6']
                    extract(model, X, y, samples, feature_names, importance_type, layers, target)
                    

        test_scores = pd.DataFrame(test_scores, index=model_names)
        generate_plots(test_scores, self.directory)
        self.save_all_scores(test_scores)

        if self.task == 'classification_binary':
            auc_fig = plt.figure()
            auc_fig.set_size_inches((10, 6))
            prc_fig = plt.figure()
            prc_fig.set_size_inches((10, 6))
            for y_test, y_pred_test, y_pred_test_scores, model_name in zip(y_test_list, y_pred_test_list,
                                                                           y_pred_test_scores_list, model_names):
                plot_roc(auc_fig, y_test, y_pred_test_scores, self.directory, label=model_name)
                plot_prc(prc_fig, y_test, y_pred_test_scores, self.directory, label=model_name)
            auc_fig.savefig(os.path.join(self.directory, 'auc_curves'))
            prc_fig.savefig(os.path.join(self.directory, 'auprc_curves'))
            
        return test_scores
    
    
    def evaluate(self, y_test, y_pred_test, y_pred_test_scores):
        if self.task == 'classification_binary':
            test_score = evaluate(y_test, y_pred_test, y_pred_test_scores)
        if self.task == 'regression':
            test_score = evaluate_regression(y_test, y_pred_test, y_pred_test_scores)
            
        return test_score
    
    
    def save_coef(self, model_list, cols):
        coef_df = pd.DataFrame(index=cols)

        dir_name = os.path.join(self.directory, 'fs')
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        for model, model_params in model_list:
            model_name = get_model_id(model_params)
            c_ = get_coef_from_model(model)
            logging.info('Saving coef...')
            model_name_col = model_name
            
            if hasattr(model, 'get_named_coef') and c_ is not None:
                file_name = os.path.join(dir_name, 'coef_' + model_name)
                coef = model.get_named_coef()
                if type(coef) == list:
                    for i, c in enumerate(coef):
                        if type(c) == pd.DataFrame:
                            c.to_csv(file_name + '_layer' + str(i) + '.csv')
                elif type(coef) == dict:
                    for c in coef.keys():
                        if type(coef[c]) == pd.DataFrame:
                            coef[c].to_csv(file_name + '_layer' + str(c) + '.csv')

            if type(c_) == list or type(c_) == tuple:
                coef_df[model_name_col] = c_[0]
            else:
                coef_df[model_name_col] = c_
                
        file_name = os.path.join(dir_name, 'coef.csv')
        coef_df.to_csv(file_name)
        
        
    def plot_coef(self, model_list):
        for model, model_name in model_list:
            plt.figure()
            file_name = os.path.join(self.directory, 'coef_' + model_name)
            for coef in model.coef_:
                plt.hist(coef, bins=20)
            plt.savefig(file_name)
            
            
    def save_all_scores(self, scores):
        file_name = os.path.join(self.directory, 'all_scores.csv')
        scores.to_csv(file_name)
        
        
    def save_score(self, data_params, model_params, score, model_name):
        file_name = os.path.join(self.directory, model_name + '_params.yml')
        logging.info("Saving yml : %s" % file_name)
        
        yml_dict = {
            'task': self.task,
            'exp_name': self.exp_name,
            'data_params': data_params,
            'pre_params': self.pre_params,
            'features_params': self.features_params,
            'model_params': model_params,
            'pipeline_params': self.pipeline_params,
            'score': str(score)}

        with open(file_name, 'w') as yaml_file:
            yaml_file.write(
                yaml.dump(yml_dict, default_flow_style=False)
            )
            
            
    def predict(self, model, x_test, y_test):
        logging.info('Predicting ...')
        y_pred_test = model.predict(x_test)
        
        if hasattr(model, 'predict_proba'):
            y_pred_test_scores = model.predict_proba(x_test)[:, 1]
        else:
            y_pred_test_scores = y_pred_test

        return y_pred_test, y_pred_test_scores
    
    
    def preprocess(self, x_train, x_test):
        logging.info('Preprocessing....')
        proc = preprocessor.get_preprocessor(self.pre_params)
        
        if proc:
            proc.fit(x_train)
            x_train = proc.transform(x_train)
            x_test = proc.transform(x_test)

            if scipy.sparse.issparse(x_train):
                x_train = x_train.todense()
                x_test = x_test.todense()
                
        return x_train, x_test
    
    
    def extract_features(self, x_train, x_test):
        if self.features_params == {}:
            return x_train, x_test
        logging.info('Feature extraction ....')

        proc = feature_extraction.get_processor(self.features_params)
        if proc:
            x_train = proc.transform(x_train)
            x_test = proc.transform(x_test)

            if scipy.sparse.issparse(x_train):
                x_train = x_train.todense()
                x_test = x_test.todense()
        return x_train, x_test
    