# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 14:00:14 2019

@author: Mahery Andriana
"""

# %%
import time
import sys
import os.path as os_path

import pandas as pd
import numpy as np
import pickle

from flask import Flask, render_template, request
from nltk import word_tokenize, RegexpTokenizer 
from sklearn.ensemble import RandomForestClassifier

# %%
app = Flask(__name__)

# %%
# finding and loading the module PATHS
PCKG_PTH = os_path.abspath(os_path.join('../'))  # necessary absolute path
PCKG_PTH_EXPLRTN = os_path.abspath(\
                    os_path.join('../exploration/'))
PCKG_PTH_PRGRMMMTN = os_path.abspath(\
                    os_path.join('../programmation/'))
PCKG_PTH_NLP = os_path.abspath(\
                    os_path.join('../natural_language_processing/'))

PCKG_PTH = [PCKG_PTH, PCKG_PTH_EXPLRTN, PCKG_PTH_PRGRMMMTN, PCKG_PTH_NLP]
print(f'module_path:')

for pckg in PCKG_PTH:
    print(f"{pckg}")
    if pckg not in sys.path: 
        sys.path.append(pckg)
print()        
print(f"sys.path:\n{sys.path}\n")

# %%
# nltk.download('punkt')
DATA = f"data/data_p6/"
PTH_DT = '../' + DATA
nan = NAN = np.nan  # WHoa!!

# %%
# Cleaning

# %%

# Random Forest Classification
n_estimators = 50
max_depth = 100

rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                            n_jobs=-1)

ovr_tagging_rf = OneVsRestTagging(df_encoded_questions_unique, N_SPLITS,
                                  matrx_tfidf, rf, 'rf')