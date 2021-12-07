# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 12:58:32 2021

@author: femiogundare
"""

import os
import sys
import logging


def set_logging(filename):
    if not os.path.exists(filename):
        os.makedirs(filename)
        
    filename = os.path.join(filename, 'log.log')
    logging.basicConfig(filename=filename, 
                        filemode='w',
                        format='%(asctime)s - {%(filename)s:%(lineno)d} - %(message)s',
                        datefmt='%m/%d %I:%M',
                        level=logging.INFO
                        )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info('Setting logs...')
    
    

debug_folder = '.'
    

class DebugFolder():

    def __init__(self, folder=None):
        self.set_debug_folder(folder)

    def get_debug_folder(self):
        global debug_folder
        return debug_folder

    def set_debug_folder(self, folder):
        if not folder is None:
            global debug_folder
            debug_folder = folder