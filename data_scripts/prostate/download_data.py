# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 12:37:52 2021

@author: femiogundare
"""

import os
import urllib3
from pathlib import Path


ROOT_DIR = str(Path(__file__).resolve().parents[2])

data_dir = ROOT_DIR + '/_database/prostate/raw_data'
processed_dir = ROOT_DIR + '/_database/prostate/processed'


def download_data():
    print ('downloading data files')
    # P1000 data
    file2 = 'https://static-content.springer.com/esm/art%3A10.1038%2Fs41588-018-0078-z/MediaObjects/41588_2018_78_MOESM6_ESM.xlsx'
    file1 = 'https://static-content.springer.com/esm/art%3A10.1038%2Fs41588-018-0078-z/MediaObjects/41588_2018_78_MOESM4_ESM.txt'
    file3 = 'https://static-content.springer.com/esm/art%3A10.1038%2Fs41588-018-0078-z/MediaObjects/41588_2018_78_MOESM10_ESM.txt'
    file4 = 'https://static-content.springer.com/esm/art%3A10.1038%2Fs41588-018-0078-z/MediaObjects/41588_2018_78_MOESM10_ESM.txt'
    file5 = 'https://static-content.springer.com/esm/art%3A10.1038%2Fs41588-018-0078-z/MediaObjects/41588_2018_78_MOESM5_ESM.xlsx'

    # Met500 files 'https://www.nature.com/articles/nature23306'

    links = [file1, file2, file3, file4, file5]

    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for link in links:
        print('Downloading file {}'.format(link))
        filename = os.path.join(data_dir, os.path.basename(link))
        with open(filename, 'wb') as f:
            """
            f.write(urllib2.urlopen(link).read())
            f.close()
            """
            http = urllib3.PoolManager()
            response = http.request('GET', link)
            data = response.data

download_data()