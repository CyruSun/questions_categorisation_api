# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 14:00:14 2019

@author: Mahery Andriana
"""

# %%
import time
import sys
import re
import os.path as os_path

import pandas as pd
import numpy as np
import logging
import pickle
import operator

from nltk import word_tokenize, RegexpTokenizer 
from matplotlib import pyplot as plt
from sklearn import model_selection as model_selection
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel

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