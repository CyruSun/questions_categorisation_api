# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 14:00:14 2019

@author: Mahery Andriana
"""

# %%
import sys
import os.path as os_path

import pandas as pd
import numpy as np

from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestClassifier

# %%
app = Flask(__name__)

# %%
# finding and loading the module PATHS
PCKG_PTH = os_path.abspath(os_path.join('../'))  # necessary absolute path
PCKG_PTH_EXPLRTN = os_path.abspath(os_path.join('../exploration/'))
PCKG_PTH_PRGRMMMTN = os_path.abspath(os_path.join('../programmation/'))
PCKG_PTH_NLP = os_path.abspath(
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
# Import modules from added path
from time_relate import timer

# %%
# nltk.download('punkt')
DATA = f"data/data_p6/"
PTH_DT = '../' + DATA
nan = NAN = np.nan  # WHoa!!


# %%
class OneVsRestTagging:
    """OnevsRest predictions."""

    from sklearn.model_selection import cross_validate

    def __init__(self, target_val, n_splits, predictors, estimator,
                 distinctive_name=None,
                 df_matching_keywords=df['Matching_keywords']):
        """
        Keywords arguments:
        target_val -- values to predict
        n_splits -- number of train/test data split for cross validation
        predictors -- explanatory variables
        estimator -- classifier
        distinctive_name -- distinctive name end of the estimator for
        attributes, features or variables
        df_matching_keywords -- dataframe rows of the intersection
        keywords between IT_tags and the question words
        """

        self.target_val = target_val
        self.n_splits = n_splits
        self.predictors = predictors
        self.estimator = estimator
        self.ls_questions_indx = list(target_val.index)
        self.ls_tags_unique = list(target_val.columns)
        # Probabilities out of the estimators predictions
        self.proba = []
        # Distinctive name  of the estimator
        self.distinctive_name = distinctive_name
        # Df of the intersection keywords between IT_tags and the
        # question
        # words: aka real tags
        self.df_matching_keywords = pd.DataFrame(df_matching_keywords)
        # Column named by the current CV run
        self.feat_name = f'High_proba_Tags_{self.distinctive_name}'
        # Initialization of the df containing the questions tags of all
        # cv tours
        self.df_questions_tags = pd.DataFrame(
            index=self.ls_questions_indx,
            columns=[self.feat_name], dtype=object)
        # Initialization of the df containing the matching of each tags
        self.df_match = pd.DataFrame(
            index=self.ls_questions_indx, columns=[self.feat_name],
            dtype=object)
        # Numbers of the intersection keywords between IT_tags and the
        # question words
        self.real_tags_numbers = []
        # Nb of tags predicted from the OVR
        self.prdct_tags_bin_numbers = []
        # ratio between matching and max tags numbers for each CV run
        self.ratios = []
        self.matching_numbers = []  # Number of matching per CV run

    def matching_ratio(self, nb_intersect_tags, nb_real_tags,
                       nb_prdct_tags_bin):
        """Return the calculated ratio.

        Keywords arguments:
        nb_intersect_tags : number of matching
        nb_real_tags : number of possible real tags
        nb_prdct_tags_bin : number of tags out of the predictions
        """

        ratio = nb_intersect_tags/max(nb_real_tags, nb_prdct_tags_bin)
        return ratio

    @timer
    def tag_questions_by_OvR_CV(self):
        """Return the tags for each questions and determining the
        matching performance of the estimator."""

        # Cross validation
        kf = model_selection.KFold(n_splits=self.n_splits)
        kf.get_n_splits(self.predictors)

        # Split data in train/test sets
        for cv_run, (train_index, test_index) in enumerate(
            kf.split(self.predictors), 1):
            X_train, X_test = self.predictors[train_index],\
            self.predictors[test_index]
            y_train, y_test = self.target_val.iloc[train_index, :],\
            self.target_val.iloc[test_index, :]

            # One vs Rest
            ovr = OneVsRestClassifier(self.estimator, n_jobs=-1).fit(
                X_train, y_train)

            # Prediction performance takes binaries values (0 or 1)
            # like the target values
            predct = ovr.predict(X_test)

            self.proba = ovr.predict_proba(X_test)
            #  print(self.proba)

            # Initialization of the matching means list between
            # predicted and true binaries values for each turn
            run_match_mean = []

            row_index = y_test.index
            # Tagging the questions
            for idx, row_ind in enumerate(row_index):
                print(f'row_ind = {row_ind}\n')
                y_test_row_ind = y_test.loc[row_ind, :]
                # print(f'y_test_row_ind =\n{y_test_row_ind}\n')
                # Faire pour chques questions

                prdct_tags_bin = predct[idx]
                # print(f'prdct_tags_bin = {prdct_tags_bin}\n')

                # Finding the matching means for each questions
                df_match_row_ind = self.df_match.loc[
                    row_ind, :] = prdct_tags_bin == y_test_row_ind
                # print(f'predct[{idx}] =\n{prdct_tags_bin}\n')
                # print(f'df_match_row_ind:\n{df_match_row_ind}\n')

                # Matching means between predicted and true binaries
                ## values for each turn
                row_match_mean = np.mean(df_match_row_ind)
                # print(f'row_match_mean =\n{row_match_mean}\n')
                run_match_mean.append(row_match_mean)

                ## Metric
                # Number of matching True
                match_true = df_match_row_ind == True
                # print(f'match_true = \n{match_true}\n')

                # Real tags set
                st_real_tags = self.df_matching_keywords.loc[row_ind,
                                                                :][0]
                # print(f'st_real_tags =\n{st_real_tags}\n')
                nb_real_tags = len(st_real_tags)
                print(f'nb_real_tags =\n{nb_real_tags}\n')
                self.real_tags_numbers.append(nb_real_tags)

                idx_tags = np.nonzero(prdct_tags_bin == 1)[0]
                # print(f'idx_tags = {idx_tags}\n')
                st_prdct_tags = {self.ls_tags_unique[
                    idx] for idx in idx_tags}
                # print(f'st_prdct_tags = {st_prdct_tags}\n')
                nb_prdct_tags_bin = np.count_nonzero(prdct_tags_bin)
                print(f'nb_prdct_tags_bin =\n{nb_prdct_tags_bin}\n')
                self.prdct_tags_bin_numbers.append(nb_prdct_tags_bin)

                # Cardinal of the intersection of the real and the
                # predicted sets

                st_intersect_tags = st_prdct_tags & st_real_tags
                print(f'st_intersect_tags = {st_intersect_tags}\n')
                nb_intersect_tags = len(st_intersect_tags)
                print(f'nb_intersect_tags = {nb_intersect_tags}\n')
                self.matching_numbers.append(nb_intersect_tags)

                ratio_row = self.matching_ratio(nb_intersect_tags,
                                                nb_real_tags,
                                           nb_prdct_tags_bin)
                # print(f'ratio_row =\n{ratio_row}\n')
                self.ratios.append(ratio_row)

                # Index of the 5 highest proba
                ind_5_max = np.argsort(self.proba[idx])[-5:]
                # print(f'ind_5_max: {ind_5_max}\n')
                # Tags with the 5 highest proba
                hi_tags = [self.ls_tags_unique[idx_max] for idx_max in\
                           ind_5_max]
                # print(f'hi_tags:\n{hi_tags}\n')
                #  and convert the column to object first in
                # order to insert a list to the cell
                self.df_questions_tags.loc[row_ind, self.feat_name] =\
                ' '.join(hi_tags)

        # print(f'real_tags_numbers =\n{self.real_tags_numbers}\n')
        # print(f'prdct_tags_bin_numbers =\n{\
        # self.prdct_tags_bin_numbers}\n')

        # Writing in the dfs
        best_tags_distinct = f'Best_Tags_{self.distinctive_name}'
        self.df_questions_tags[best_tags_distinct] =\
        self.df_questions_tags[self.feat_name].str.split(' ')
        self.df_questions_tags[
            f'Matching_Numbers_{self.distinctive_name}'] =\
        self.matching_numbers
        self.df_questions_tags[
            f'Assessment_Ratio_{self.distinctive_name}'] = self.ratios

        # Conversion to set
        self.df_questions_tags[
            f'Best_Unique_Tags_{self.distinctive_name}'] =\
        self.df_questions_tags.apply(
            lambda row: set(row[best_tags_distinct]), axis=1)
        return self.df_questions_tags, self.matching_numbers,\
    self.ratios
# %%
@app.route('/')  # décorateur permets d'entrer des métadonnées
def home():
    return render_template('pages/home.html')

# %%
def tag_question():
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
    df_questions_tags_rf, matching_numbers_rf,\
    ratios_rf = ovr_tagging_rf.tag_questions_by_OvR_CV()


# %%
# Fonctionnement pour exécution en ligne de commande
if __name__ == '__main__':
    app.run(debug=True, port=5000)
