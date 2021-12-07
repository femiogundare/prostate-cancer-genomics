# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 13:20:24 2021

@author: femiogundare
"""

import os


BASE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(BASE_PATH, '_database')
GENE_PATH = os.path.join(DATA_PATH, 'genes')
PATHWAY_PATH = os.path.join(DATA_PATH, 'pathways')
PROSTATE_DATA_PATH = os.path.join(DATA_PATH, 'prostate')
REACTOME_PATHWAY_PATH = os.path.join(PATHWAY_PATH, 'Reactome')

RUN_PATH = os.path.join(BASE_PATH, 'train')

LOG_PATH = os.path.join(BASE_PATH, '_logs')
PROSTATE_LOG_PATH = os.path.join(LOG_PATH, 'prostate')

PARAMS_PATH = os.path.join(RUN_PATH, 'params')
PROSTATE_PARAMS_PATH = os.path.join(PARAMS_PATH, 'prostate')

PLOTS_PATH = os.path.join(BASE_PATH, '_plots')