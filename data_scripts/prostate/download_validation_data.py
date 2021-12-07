# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 17:56:57 2021

@author: femiogundare
"""

import os
import urllib3
from pathlib import Path


ROOT_DIR = str(Path(__file__).resolve().parents[2])

data_dir = ROOT_DIR + '/_database/prostate/external_validation'
processed_dir = ROOT_DIR + '/_database/prostate/processed'


if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    
    
def download_data_MET500():
    met500_dir = os.path.join(data_dir, 'MET500')
    if not os.path.exists(met500_dir):
        os.makedirs(met500_dir)
        
    print('Downloading data files')
    # sub_dir = 'Met500'
    # file1= 'https://met500.path.med.umich.edu/met500_download_datasets/cnv_v4.csv'
    file1 = 'https://met500.path.med.umich.edu/met500_download_datasets/somatic_v4.csv'
    file2 = 'https://www.dropbox.com/s/62fqw2zgc6ayxvg/Met500_cnv.txt?dl=0'
    file3 = 'https://www.dropbox.com/s/htcx4f09k231l5m/samples.txt?dl=0'

    links = [file1, file2, file3]

    for link in links:
        print ('downloading file {}'.format(link))
        filename = os.path.join(met500_dir, os.path.basename(link))
        with open(filename, 'wb') as f:
            """
            f.write(urllib2.urlopen(link).read())
            f.close()
            """
            http = urllib3.PoolManager()
            response = http.request('GET', link)
            data = response.data
            
            
            
def download_data_PRAD():
    # https://www.nature.com/articles/nature20788#MOESM323
    file1 = 'https://static-content.springer.com/esm/art%3A10.1038%2Fnature20788/MediaObjects/41586_2017_BFnature20788_MOESM324_ESM.zip'
    file2 = 'https://static-content.springer.com/esm/art%3A10.1038%2Fnature20788/MediaObjects/41586_2017_BFnature20788_MOESM325_ESM.zip'
    links = [file1, file2]
    for link in links:
        print ('downloading file {}'.format(link))
        filename = os.path.join(data_dir, os.path.basename(link))
        with open(filename, 'wb') as f:
            """
            f.write(urllib2.urlopen(link).read())
            f.close()
            """
            http = urllib3.PoolManager()
            response = http.request('GET', link)
            data = response.data
            
            
download_data_MET500()
download_data_PRAD()
print('Done')