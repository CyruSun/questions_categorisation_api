# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 21:01:45 2018

@author: Mahery
"""

import sys

import numpy as np
import os.path as os_path


# finding and loading the module PATHS
PCKG_PTH_EXPLRTN = os_path.abspath(os_path.join('../exploration/'))
PCKG_PTH_PRGRMMMTN = os_path.abspath(os_path.join('../programmation/'))
PCKG_PTH_NLP = os_path.abspath(
                    os_path.join('../natural_language_processing/'))

ls_path = {PCKG_PTH_EXPLRTN, PCKG_PTH_PRGRMMMTN, PCKG_PTH_NLP}
print(f'module_path:')

for pckg in ls_path:
    print(f"{pckg}")
    if pckg not in sys.path:
        sys.path.append(pckg)
print()
print(f"sys.path:\n{sys.path}\n")

from time_relate import timer

NAN = np.nan


def set_range(lst):
    """Convertit un range de liste en un ensemble pour que les tests
    d'appartenance soit en O(1)."""
    return set(range(lst))


def zip_to_dict(lst_keys, lst_val):
    """Create a dictionnary from keys and values lists.

    Keywords arguments:
        lst_keys -- liste des clés
        lst_val -- liste des valeurs
    """
    zp = zip(lst_keys, lst_val)  # regroupement sous forme de tuples
    return dict(zp)


def reshape_one_feat(arr):
    """Reshape single featured data either using array.reshape(-1, 1)."""
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def reshape_one_sample(arr):
    """Reshape single sampled data either using array.reshape(1, -1)."""
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def keys_from_dct_val(dct, key):
    """Returns index of key in a dict."""
    lst_ks = list(dct.keys())  # Keys
    return [lst_ks.index(key)]


def flatten_list(lst):
    """Flatten a list by returning all the values in a 1d list."""
    return [y for x in lst for y in x]


def complete_lst_with_term(lst, lngth, cmplt_term=NAN):
    """Complete a list with a term until a certain length."""
    while len(lst) <= lngth:
        lst.append(cmplt_term)
    return lst


@timer
def push_to_unique_column(df_with_null_values):
    """Push inside an unique column of a 1 and NAN value dataframe, all
    the 1 values from all the same features names columns.

    Keywords arg:
    df_with_null_values -- DataFrame with 1 and NAN values."""

    for column in df_with_null_values.columns:
        # print('>' + 34 * '-' + '<>==O==<>' + 34 * '-' + '<')
        print(f'column:{column}\n')
        df_column = df_with_null_values[column]
        columns_number = df_column.shape

        # Testing if many columns with the same name exist
        if len(columns_number) > 1:
            same_name_columns_number = df_column.shape[1]

            # Initialization where 0 is replaced by NAN
            rslt = df_column.iloc[:, 0].replace(0, NAN)

            for i in range(1, same_name_columns_number):
                # Columns values pushed in the only first column, and
                # zeros are replaced by NAN to save memory
                rslt = rslt.combine_first(df_column.iloc[:, i])
                rslt.replace(0, NAN, inplace=True)

            # Replacing all the columns
            del df_with_null_values[column]
            df_with_null_values[column] = rslt
    return df_with_null_values

# def dtfrm_string_to_columns_by_length(dtfrm, column_nm, lst_unwanted=[]):
#    """Assignment of description string elements to new columns
#    by their length.
#
#    DOES NOT WORK when there are mulitple characters with the same length
#    in a string!
#
#    Keywords arguments:
#        lst_unwanted -- list of unwanted elements
#    """
#    start_time = time.time()  # Running timing
#    for dtfrm_row in dtfrm[column_nm]:
#        for n, strng in enumerate(dtfrm_row):
#            if strng not in lst_unwanted:
#                # length of the string
#                ln_str = len(strng)
#                # Assignment
#                dtfrm.loc[n, f'{column_nm}_{ln_str}'] = strng
#    print("--- %s seconds ---" % (time.time() - start_time))
#    return dtfrm
