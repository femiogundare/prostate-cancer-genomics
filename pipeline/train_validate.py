# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 14:00:01 2021

@author: femiogundare
"""

import os
import logging
import numpy as  np
import pandas as pd
import scipy
import yaml
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn import decomposition
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from data_scripts.data_access import Data
from model.model_factory import get_model
from preprocessing import preprocessor
from utilities.evaluate import evaluate
from utilities.plots import plot_confusion_matrix
from utilities.random_seed import set_random_seeds



def plot_2D(x, y, keys, marker='o'):
    classes = np.unique(y)
    for c in classes:
        plt.scatter(x[y == c, 0], x[y == c, 1], marker=marker)
    plt.legend(keys)
    
    
def get_validation_primary(cols, cnv_split):
    current_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    validation_data_dir = os.path.join(current_dir, '_database/prostate/external_validation/')

    valid_cnv = pd.read_csv(os.path.join(validation_data_dir, 'PRAD/cnv_matrix.csv'), index_col=0)
    valid_mut = pd.read_csv(os.path.join(validation_data_dir, 'PRAD/mut_matrix.csv'), index_col=0)

    genes = cols.get_level_values(0).unique()
    genes_df = pd.DataFrame(index=genes)

    valid_mut_df = genes_df.merge(valid_mut.T, how='left', left_index=True, right_index=True).T
    valid_cnv_df = genes_df.merge(valid_cnv.T, how='left', left_index=True, right_index=True).T

    df_list = [valid_mut_df, valid_cnv_df]
    data_type_list = ['gene_final', 'cnv']

    if cnv_split:
        valid_cnv_ampl = valid_cnv_df.copy()
        valid_cnv_ampl[valid_cnv_ampl <= 0.0] = 0.
        valid_cnv_ampl[valid_cnv_ampl > 0.0] = 1.0

        valid_cnv_del = valid_cnv_df.copy()

        valid_cnv_del[valid_cnv_del >= 0.0] = 0.
        valid_cnv_del[valid_cnv_del < 0.0] = 1.0
        df_list = [valid_mut_df, valid_cnv_del, valid_cnv_ampl]
        data_type_list = ['mut', 'cnv_del', 'cnv_amp']

    all_data = pd.concat(df_list, keys=data_type_list, join='inner', axis=1, )
    all_data = all_data.swaplevel(i=0, j=1, axis=1)

    # order the columns based on genes
    order = all_data.columns.levels[0]
    all_data = all_data.reindex(columns=order, level=0)
    all_data.fillna(0, inplace=True)
    x = all_data.as_matrix()
    samples = pd.DataFrame(index=all_data.index)
    response = np.zeros((x.shape[0],))
    genes = all_data.columns
    
    return x, samples, response, genes



def get_validation_metastatic(cols, cnv_split):
    common_samples = ['MO_1008',
                      'MO_1012',
                      'MO_1013',
                      'MO_1014',
                      'MO_1015',
                      'MO_1020',
                      'MO_1040',
                      'MO_1074',
                      'MO_1084',
                      'MO_1094',
                      'MO_1095',
                      'MO_1096',
                      'MO_1114',
                      'MO_1118',
                      'MO_1124',
                      'MO_1128',
                      'MO_1130',
                      'MO_1132',
                      'MO_1139',
                      'MO_1161',
                      'MO_1162',
                      'MO_1176',
                      'MO_1179',
                      'MO_1184',
                      'MO_1192',
                      'MO_1202',
                      'MO_1215',
                      'MO_1219',
                      'MO_1232',
                      'MO_1241',
                      'MO_1244',
                      'MO_1249',
                      'MO_1262',
                      'MO_1277',
                      'MO_1316',
                      'MO_1337',
                      'MO_1339',
                      'MO_1410',
                      'MO_1421',
                      'MO_1447',
                      'MO_1460',
                      'MO_1473',
                      'TP_2001',
                      'TP_2010',
                      'TP_2020',
                      'TP_2032',
                      'TP_2034',
                      'TP_2054',
                      'TP_2060',
                      'TP_2061',
                      'TP_2064',
                      'TP_2069',
                      'TP_2077',
                      'TP_2078',
                      'TP_2079']

    prostate_samples = ['MO_1008', 'MO_1012', 'MO_1013', 'MO_1014', 'MO_1015', 'MO_1020', 'MO_1040', 'MO_1066',
                        'MO_1074', 'MO_1084',
                        'MO_1093', 'MO_1094', 'MO_1095', 'MO_1096', 'MO_1112', 'MO_1114', 'MO_1118', 'MO_1124',
                        'MO_1128', 'MO_1130',
                        'MO_1132', 'MO_1139', 'MO_1161', 'MO_1162', 'MO_1176', 'MO_1179', 'MO_1184', 'MO_1192',
                        'MO_1200', 'MO_1201',
                        'MO_1202', 'MO_1214', 'MO_1215', 'MO_1219', 'MO_1221', 'MO_1232', 'MO_1240', 'MO_1241',
                        'MO_1244', 'MO_1249',
                        'MO_1260', 'MO_1262', 'MO_1263', 'MO_1277', 'MO_1307', 'MO_1316', 'MO_1336', 'MO_1337',
                        'MO_1339', 'MO_1410',
                        'MO_1420', 'MO_1421', 'MO_1437', 'MO_1443', 'MO_1446', 'MO_1447', 'MO_1460', 'MO_1469',
                        'MO_1472', 'MO_1473',
                        'MO_1482', 'MO_1490', 'MO_1492', 'MO_1496', 'MO_1499', 'MO_1510', 'MO_1511', 'MO_1514',
                        'MO_1517', 'MO_1541',
                        'MO_1543', 'MO_1553', 'MO_1556', 'TP_2001', 'TP_2009', 'TP_2010', 'TP_2020', 'TP_2032',
                        'TP_2034', 'TP_2037',
                        'TP_2043', 'TP_2054', 'TP_2060', 'TP_2061', 'TP_2064', 'TP_2069', 'TP_2077', 'TP_2078',
                        'TP_2079', 'TP_2080',
                        'TP_2081', 'TP_2090', 'TP_2093', 'TP_2096', 'TP_2156']

    met500_samples = set(prostate_samples).difference(common_samples)
    common_samples = pd.DataFrame(index=met500_samples)
    current_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    validation_data_dir = os.path.join(current_dir, '_database/prostate/external_validation/')

    valid_cnv = pd.read_csv(os.path.join(validation_data_dir, 'Met500/Met500_cnv.txt'), index_col=0, sep='\t')
    valid_mut = pd.read_csv(os.path.join(validation_data_dir, 'Met500/Met500_mut_matrix.csv'), index_col=0)

    valid_cnv = valid_cnv.T
    valid_cnv[valid_cnv > 1.] = 1.
    valid_cnv[valid_cnv < 0.] = -1.
    valid_mut.index = valid_mut.index.str.split('.', 1).str[0]

    genes = cols.get_level_values(0).unique()
    genes_df = pd.DataFrame(index=genes)

    valid_mut_df = common_samples.merge(valid_mut, how='inner', left_index=True, right_index=True)
    valid_cnv_df = common_samples.merge(valid_cnv, how='inner', left_index=True, right_index=True)

    valid_mut_df = genes_df.merge(valid_mut_df.T, how='left', left_index=True, right_index=True).T
    valid_cnv_df = genes_df.merge(valid_cnv_df.T, how='left', left_index=True, right_index=True).T

    df_list = [valid_mut_df, valid_cnv_df]
    data_type_list = ['gene_final', 'cnv']

    if cnv_split:
        valid_cnv_ampl = valid_cnv_df.copy()
        valid_cnv_ampl[valid_cnv_ampl <= 0.0] = 0.
        valid_cnv_ampl[valid_cnv_ampl > 0.0] = 1.0

        valid_cnv_del = valid_cnv_df.copy()

        valid_cnv_del[valid_cnv_del >= 0.0] = 0.
        valid_cnv_del[valid_cnv_del < 0.0] = 1.0
        df_list = [valid_mut_df, valid_cnv_del, valid_cnv_ampl]
        data_type_list = ['important_mutations', 'cnv_deletion', 'cnv_amplification']

    all_data = pd.concat(df_list, keys=data_type_list, join='inner', axis=1, )
    all_data = all_data.swaplevel(i=0, j=1, axis=1)

    # order the columns based on genes
    order = all_data.columns.levels[0]
    all_data = all_data.reindex(columns=order, level=0)
    all_data.fillna(0, inplace=True)

    x = all_data.values
    samples = pd.DataFrame(index=all_data.index)
    response = np.ones((x.shape[0],))
    genes = all_data.columns

    return x, samples, response, genes



class TrainValidatePipeline:
    def __init__(self, data_params, pre_params, feature_params, model_params, pipeline_params, exp_name):

        self.data_params = data_params
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
            
        self.prepare_saving_dir()

    def prepare_saving_dir(self):
        self.directory = self.exp_name
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
            
    def save_prediction(self, info, y_pred, y_pred_score, y_test, model_name, training=False):

        if training:
            file_name = os.path.join(self.directory, model_name + '_training.csv')
        else:
            file_name = os.path.join(self.directory, model_name + '_testing.csv')
            
        logging.info("saving : %s" % file_name)
        
        info['pred'] = y_pred
        info['score'] = y_pred_score
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
    
    def run(self):
        
        logging.info('loading data....')
        
        data = Data(**self.data_params[0])
        # Use the whole dataset for training
        x_train, samples_train, response_train, genes_train = data.get_data()

        data_types = genes_train.get_level_values(1).unique()
        
        if len(data_types) > 2:
            cnv_split = True
        else:
            cnv_split = False

        # divide the training set into two blanced training sets. Later we will train 2 models and combine their outputs

        index_pos = np.where(response_train == 1)[0]
        index_neg = np.where(response_train == 0)[0]
        n_pos = index_pos.shape[0]
        # select the same number of samples as the positive class
        index_neg1 = index_neg[0:n_pos]

        x_train_pos = x_train[index_pos, :]
        x_train_neg = x_train[index_neg1, :]
        x_train1 = np.concatenate((x_train_pos, x_train_neg))

        y_train_pos = response_train[index_pos, :]
        y_train_neg = response_train[index_neg1, :]
        y_train1 = np.concatenate((y_train_pos, y_train_neg))

        samples_train_pos = samples_train[index_pos]
        samples_train_neg1 = samples_train[index_neg1]
        samples_train1 = np.concatenate((samples_train_pos, samples_train_neg1))

        # second training set
        index_neg2 = index_neg[n_pos:]
        x_train_neg2 = x_train[index_neg2, :]
        x_train2 = np.concatenate((x_train_pos, x_train_neg2))

        y_train_neg2 = response_train[index_neg2, :]
        y_train2 = np.concatenate((y_train_pos, y_train_neg2))

        samples_train_pos = samples_train[index_pos]
        samples_train_neg2 = samples_train[index_neg2]
        samples_train2 = np.concatenate((samples_train_pos, samples_train_neg2))

        # Get test dataset (from external validation)
        x_test_mets, samples_test_mets, response_test_mets, genes_test_mets = get_validation_metastatic(genes_train, cnv_split)
        
        x_test_primary, samples_test_primary, response_test_primary, genes_test_primary = get_validation_metastatic(genes_train, 
                                                                                                                    cnv_split
                                                                                                                    )
        

        # pre-processing
        logging.info('Preprocessing....')
        _, x_test_mets = self.preprocess(x_train, x_test_mets)
        _, x_test_primary = self.preprocess(x_train, x_test_primary)
        _, x_train1 = self.preprocess(x_train, x_train1)
        _, x_train2 = self.preprocess(x_train, x_train2)


        test_scores = []
        
        fig = plt.figure()
        fig.set_size_inches((10, 6))
        pred_scores = []
        
        if type(self.model_params) == list:
            for m in self.model_params:
                # get model
                set_random_seeds(random_seed=20080808)

                model1 = get_model(m)
                model2 = get_model(m)
                logging.info('Fitting...')

                model1 = model1.fit(x_train1, y_train1)
                model2 = model2.fit(x_train2, y_train2)

                logging.info('Predicting...')

                def predict(x_test, y_test, info_test, model_name, test_set_name):
                    pred = {}
                    y_pred_test2, y_pred_test_scores2 = self.predict(model2, x_test, y_test)
                    y_pred_test1, y_pred_test_scores1 = self.predict(model1, x_test, y_test)

                    y_pred_test_scores = (y_pred_test_scores1 + y_pred_test_scores2) / 2.
                    y_pred_test = y_pred_test_scores > 0.5

                    logging.info('Scoring ...')
                    test_score = evaluate(y_test, y_pred_test, y_pred_test_scores)
                    cnf_matrix = confusion_matrix(y_test, y_pred_test)

                    pred['model'] = model_name
                    pred['data_set'] = test_set_name
                    pred = dict(pred, **test_score)
                    pred_scores.append(pred)

                    logging.info('Saving results...')

                    model_name = model_name + '_' + test_set_name
                    self.save_score(test_score, model_name)
                    self.save_prediction(info_test, y_pred_test, y_pred_test_scores, y_test, model_name)
                    self.save_cnf_matrix([cnf_matrix], [model_name])

                if 'id' in m:
                    model_name = m['id']
                else:
                    model_name = m['type']

                predict(x_test_mets, response_test_mets, samples_test_mets, model_name, '_mets')
                predict(x_test_primary, response_test_primary, samples_test_primary, model_name, '_primary')

                pred_scores_df = pd.DataFrame(pred_scores)
                pred_scores_df.to_csv(os.path.join(self.directory, 'testing_scores.csv'))

        return test_scores
    
    
    
    def save_layer_outputs(self, x_train_layer_outputs, y_train, y_train_pred, x_test_layer_outputs, y_test):
        fig = plt.figure(1, figsize=(10, 9))
        for i, (x_train, x_test) in enumerate(zip(x_train_layer_outputs[:-2], x_test_layer_outputs[:-2])):
            
            pca = decomposition.PCA(n_components=50)
            X_embedded_train = pca.fit_transform(x_train[0])
            X_embedded_test = pca.transform(x_test[0])
    
            X_embedded = np.concatenate((X_embedded_train, X_embedded_test))
            tsne = TSNE(n_components=2)
            X_embedded = tsne.fit_transform(X_embedded)
            n = X_embedded_train.shape[0]
            X_embedded_train = X_embedded[0:n, :]
            X_embedded_test = X_embedded[n:, :]
    
            # print X_embedded.shape, y.shape
            plt.figure(figsize=(10, 9))
            plot_2D(X_embedded_train, y_train[:, 0], ['Primary', 'Metastatic'], 'o')
            plot_2D(X_embedded_test, y_test[:, 0], ['Primary', 'Metastatic'], 'X')
    
            # https://stackoverflow.com/questions/37718347/plotting-decision-boundary-for-high-dimension-data
            # create meshgrid
            resolution = 100  # 100x100 background pixels
            X2d_xmin, X2d_xmax = np.min(X_embedded_train[:, 0]), np.max(X_embedded_train[:, 0])
            X2d_ymin, X2d_ymax = np.min(X_embedded_train[:, 1]), np.max(X_embedded_train[:, 1])
            xx, yy = np.meshgrid(np.linspace(X2d_xmin, X2d_xmax, resolution),
                                 np.linspace(X2d_ymin, X2d_ymax, resolution))
    
            # approximate Voronoi tesselation on resolution x resolution grid using 1-NN
            background_model = KNeighborsClassifier(n_neighbors=1).fit(X_embedded_train, y_train_pred)
            voronoiBackground = background_model.predict(np.c_[xx.ravel(), yy.ravel()])
            voronoiBackground = voronoiBackground.reshape((resolution, resolution))
    
            # plot
            cmap = plt.get_cmap('jet')
            plt.contourf(xx, yy, voronoiBackground, cmap=cmap, alpha=.1)
    
            file_name = os.path.join(self.directory, 'layer_output_' + str(i))
            plt.savefig(file_name)
            plt.close()
        
        
    def save_cnf_matrix(self, cnf_matrix_list, model_list):
        for cnf_matrix, model in zip(cnf_matrix_list, model_list):
            plt.figure()
            plot_confusion_matrix(cnf_matrix, classes=['Primary', 'Metastatic'],
                                  title='Confusion matrix, without normalization')
            file_name = os.path.join(self.directory, 'confusion_' + model)
            plt.savefig(file_name)

            plt.figure()
            plot_confusion_matrix(cnf_matrix, normalize=True, classes=['Primary', 'Metastatic'],
                                  title='Normalized confusion matrix')
            file_name = os.path.join(self.directory, 'confusion_normalized_' + model)
            plt.savefig(file_name)
            
            
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
        

    def save_score(self, score, model_name):
        file_name = os.path.join(self.directory, model_name + '_params.yml')
        logging.info("Saving yml : {}".format(file_name))
        with open(file_name, 'w') as yaml_file:
            yaml_file.write(
                yaml.dump([self.data_params, self.model_params, self.pre_params, str(score)], default_flow_style=False))
            

    def predict(self, model, x_test, y_test):
        logging.info('Predicitng ...')
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
    
    
    """
    def extract_features(self, x_train, x_test):
        if self.features_params == {}:
            return x_train, x_test
        logging.info('feature extraction ....')

        proc = feature_extraction.get_processor(self.features_params)
        if proc:
            # proc.fit(x_train)
            x_train = proc.transform(x_train)
            x_test = proc.transform(x_test)

            if scipy.sparse.issparse(x_train):
                x_train = x_train.todense()
                x_test = x_test.todense()
        return x_train, x_test
    """