# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 20:51:10 2021

@author: femiogundare
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT_DIR = str(Path(__file__).resolve().parents[2])

input_dir = ROOT_DIR + '/_database/prostate/raw_data'
output_dir = ROOT_DIR + '/_database/prostate/splits'



def get_response():
    filename = '41588_2018_78_MOESM5_ESM.xlsx'
    df = pd.read_excel(os.path.join(input_dir, filename), sheet_name='Supplementary_Table3.txt', skiprows=2)
    response = pd.DataFrame()
    response['id'] = df['Patient.ID']
    response['response'] = df['Sample.Type']
    response['response'] = response['response'].replace('Metastasis', 1)
    response['response'] = response['response'].replace('Primary', 0)
    response = response.drop_duplicates()
    
    return response


response = get_response()
all_ids = response
ids = response['id'].values
y = response['response'].values

ids_train, ids_test, y_train, y_test = train_test_split(ids, y, test_size=0.1, stratify=y, random_state=422342)
ids_train, ids_validate, y_train, y_validate = train_test_split(ids_train, y_train, test_size=len(y_test),
                                                                stratify=y_train, random_state=422342)

test_set = pd.DataFrame({'id': ids_test, 'response': y_test})
train_set = pd.DataFrame({'id': ids_train, 'response': y_train})
validate_set = pd.DataFrame({'id': ids_validate, 'response': y_validate})

test_set.to_csv(os.path.join(output_dir, 'test_set.csv'))
validate_set.to_csv(os.path.join(output_dir, 'validation_set.csv'))
train_set.to_csv(os.path.join(output_dir, 'training_set.csv'))

total_number_samples = train_set.shape[0]
number_patients = np.geomspace(100, total_number_samples, 20)
number_patients = [int(s) for s in number_patients][::-1]



for i, n, in enumerate(number_patients):
    if i == 0:
        filename = os.path.join(output_dir, 'training_set_0.csv')
        train_set.to_csv(filename)
        continue
    number_samples = n
    ids_train, ids_validate, y_train, y_validate = train_test_split(ids_train, y_train, train_size=n,
                                                                    stratify=y_train, random_state=422342)
    train_set = pd.DataFrame({'id': ids_train, 'response': y_train})
    filename = os.path.join(output_dir, 'training_set_{}.csv'.format(i))
    train_set.to_csv(filename)