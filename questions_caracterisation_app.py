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
import pickle

from nltk import RegexpTokenizer
from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection as model_selection
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# %%
app = Flask(__name__)

# %%
# finding and loading the module PATHS
PCKG_PTH = os_path.abspath(os_path.join('./lib'))  # necessary absolute path

print(f'module_path:')
if PCKG_PTH not in sys.path:
    sys.path.append(PCKG_PTH)
print()
print(f"sys.path:\n{sys.path}\n")

# %%
# Import modules from added path
from time_relate import timer
import morphing as mrphng
import data_cleaning_categorisation as clean

# %%
# nltk.download('punkt')
PATH_DATA = './data/'
nan = NAN = np.nan  # WHoa!!
TOKENIZER = RegexpTokenizer(r'[\w+(?=+)]*')


# %%
class OneVsRestTagging:
    """OnevsRest predictions."""

    from sklearn.model_selection import cross_validate

    def __init__(self, df_target_val, n_splits, predictors, estimator):
        """
        Keywords arguments:
        df_target_val -- df of values to predict
        n_splits -- number of train/test data split for cross validation
        predictors -- explanatory variables
        estimator -- classifier
        last_ind -- index of input question at the last df row
        """

        self.target_val = df_target_val
        self.n_splits = n_splits
        self.predictors = predictors
        self.estimator = estimator
        self.ls_tags_unique = list(df_target_val.columns)

        # Probabilities out of the estimators predictions
        self.proba = []

        self.last_ind = df_target_val.index[-1] + 1

    @timer
    def tag_questions_OvR(self):
        """Return the tags for each questions and determining the
        matching performance of the estimator."""

        # Partition
        kf = model_selection.KFold(n_splits=self.n_splits)
        # Extracting the last train/test indexes
        fold_indexes = [(train_index, test_index) for train_index,
                        test_index in kf.split(self.predictors)]

        # Split data in train/test sets
        last_train_index = fold_indexes[self.n_splits-1][0]
        last_test_index = fold_indexes[self.n_splits-1][1]
        X_train, X_test = self.predictors[last_train_index],\
            self.predictors[last_test_index]
        y_train, _ = self.target_val.iloc[last_train_index, :],\
            self.target_val.iloc[last_test_index, :]

        # One vs Rest
        ovr = OneVsRestClassifier(self.estimator, n_jobs=-1).fit(X_train,
                                                                 y_train)
        self.proba = ovr.predict_proba(X_test)

        # Tagging the question
        # Index of the 5 highest proba
        ind_5_max = np.argsort(self.proba[-1])[-5:]
        # Tags with the 5 highest proba
        hi_tags = [self.ls_tags_unique[idx_max] for idx_max in ind_5_max]
        return set(hi_tags)

# %%
@app.route('/')
def home():
    home_title = 'Stackoverflow questions tagging API'
    abstract = 'API that returns 5 tags onto a question. The inputs are the\
    title and the body of the question.'
    return render_template('home.html', home_title=home_title,
                           abstract=abstract)
# %%
@app.route('/form')
def form():
    form_title = 'form of the question'
    nb_cols = 80
    return render_template('form.html', form_title=form_title, nb_cols=nb_cols)

# %%
@app.route('/result_tags')
def result_tags():
    return render_template('result_tags.html')

# %%
@timer
def binarize_df(df_complete, tokenizer=TOKENIZER):
    """Binarization of a dataframe.
    Keyword arguments:
    df_complete -- dataframe to encode
    """
    # loading
    with open(PATH_DATA + 'unwanted.pkl', 'rb') as p:
        unwanted = pickle.load(p)
    with open(PATH_DATA + 'st_match_words.pkl', 'rb') as p:
        st_match_words = pickle.load(p)
    with open(PATH_DATA + 'st_matching_words_2.pkl', 'rb') as p:
        st_matching_words_2 = pickle.load(p)
    # Join the Title and Body strings
    df_complete['Joint_Question_Words'] = df_complete[
        'Title'] + " " + df_complete['Body'].map(str)
    # Delete the html markups
    df_complete['Joint_Question_Words'] = df_complete[
        'Joint_Question_Words'].apply(
        lambda body_str: clean.strip_tags(body_str))

    # The all keywords
    df_complete['Question_Words'] = df_complete[
        'Joint_Question_Words'].apply(tokenizer.tokenize)
    # Delete the unwanted elements
    df_complete['Question_Words'] = df_complete['Question_Words'].apply(
            lambda x: {elmnt for elmnt in set(x) if elmnt not in unwanted})
    # Convert to set
    df_complete['Question_Words'] = df_complete.apply(
        lambda row: set(row['Question_Words']), axis=1)

    # Find the matching keywords
    df_complete['Matching_keywords'] = df_complete['Question_Words'].apply(
            lambda st: {kwrd for kwrd in st if kwrd in st_match_words})

    # Initialization of the df cotaining all the tags built for each questions
    df_questions = pd.DataFrame()
    # Relevant words
    df_questions['Matching_Words'] = df_complete[
        'Matching_keywords'].apply(lambda st: {
            kwrd for kwrd in st if kwrd in st_matching_words_2})
    df_questions['Joint_Matching_Words'] = df_questions[
            'Matching_Words'].apply(" ".join)

    # One column per tag
    df_questions_words = df_questions['Joint_Matching_Words'].str.split(
            ' ', expand=True)
    # Selection of the str value that replaces none type
    replace_none = 'replace_NONE'
    df_questions_words.fillna(value=replace_none, inplace=True)

    questions_columns = list(df_questions_words.columns)
    # Encoded dataframe for every tags of every features
    lst_encoded_questions = [pd.get_dummies(
            df_questions_words[feat]) for feat in questions_columns]
    # Binary dataframe of all tags
    df_encoded_questions = pd.concat(lst_encoded_questions, axis=1,
                                     sort='True')

    # Tags to ignore
    col_to_del = ['replace_NONE', '', ' ']
    # Delete all empty tags columns
    for col in col_to_del:
        try:
            df_encoded_questions.drop(col, axis=1, inplace=True)
        except KeyError:
            print(f"The '{col}' column doesn't exist.\n")
        finally:
            print("proceed")
    df_encoded_unique = mrphng.push_to_unique_column(
            df_encoded_questions)
    # Binarization without NaN values
    df_encoded_unique.fillna(value=int(0), inplace=True)
    return df_encoded_unique


# %%
@app.route('/send', methods=['GET', 'POST'])
def send(tokenizer=TOKENIZER):
    if request.method == 'POST':
        # loading
        with open(PATH_DATA + 'df.pkl', 'rb') as p:
            df = pickle.load(p)
        with open(PATH_DATA + 'tfidf_stopwords.pkl', 'rb') as p:
            tfidf_stopwords = pickle.load(p)

        df_encoded_questions_unique = pd.read_csv(
                PATH_DATA + 'df_encoded_questions_unique.csv', sep='\t',
                low_memory=False)
        df_encoded_questions_unique.rename(index=df_encoded_questions_unique[
                'Unnamed: 0'], inplace=True)
        # Insert original row indexes
        df_encoded_questions_unique.rename(index=df_encoded_questions_unique[
            'Unnamed: 0'], inplace=True)
        del(df_encoded_questions_unique['Unnamed: 0'])
        columns = list(df.columns)
        add_ind = df.index[-1] + 1
        q_title = request.form['q_title']
        q_body = request.form['q_body']

        # Safety conversions
        q_title = str(q_title)
        q_body = str(q_body)

        df_input = pd.DataFrame(
                np.array([[q_title, {}, q_body, NAN, {}, {}, {}]]),
                index=[add_ind], columns=columns)
        df_complete = pd.concat([df, df_input])
        # All characters

        # Replace the NAN float type values
        df_complete.replace(NAN, '', regex=True, inplace=True)
        # Convert to list
        ls_matching = df_complete.apply(lambda row: list(
                row['Matching_keywords']), axis=1).values.tolist()

        df_input_encoded = binarize_df(df_input)
        df_encoded_complete = pd.concat(
                [df_encoded_questions_unique, df_input_encoded],
                axis=0, sort=True)
        df_encoded_complete.fillna(value=int(0), inplace=True)
        # tf-idf
        ls_joint_matching = [" ".join(ls) for ls in ls_matching]
        tfidf = TfidfVectorizer(tokenizer=tokenizer.tokenize,
                                stop_words=tfidf_stopwords)
        matrx_tfidf = tfidf.fit_transform(ls_joint_matching)
        # Random forest classification
        N_ESTIMATORS = 50
        MAX_DEPTH = 50
        N_SPLITS = 5
        rf = RandomForestClassifier(n_estimators=N_ESTIMATORS, n_jobs=-1,
                                    max_depth=MAX_DEPTH)
        ovr_tagging_rf = OneVsRestTagging(df_encoded_complete, N_SPLITS,
                                          matrx_tfidf, rf)
        hi_tags = ovr_tagging_rf.tag_questions_OvR()

        return render_template('result_tags.html', tags=hi_tags)
    return render_template('form.html')


# %%
# CLI execution
if __name__ == '__main__':
    app.run(debug=True, port=5000)
