# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 15:21:50 2018

@author: Mahery
"""
import data_exploration as data_exp
import pandas as pd

from IPython.display import display as disp
from numpy import transpose as trans
from html.parser import HTMLParser


class MLStripper(HTMLParser):
    """Stripping the HTML tags, created by Eloff."""
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def get_data(self):
        return ''.join(self.fed)


def strip_tags(html):
    """functions to call the HTML tags stripper, created by Eloff."""
    s = MLStripper()
    s.feed(html)
    return s.get_data()


def doublons(dtfrm: pd.DataFrame, index_name: str):
    """Retourne les valeurs doublées d'une colonnes d'une dataframe,
    ainsi que les indices des doublons regroupés par l'index.

    Keyword arguments:
    dtfrm -- dataframe
    index_name -- index de la colonne (str)
    """
    # dataframe des doublons
    dtfrm_duplications = dtfrm[dtfrm.duplicated(subset=index_name, keep=False)]

    # Affiche toutes les lignes et colonnes
    print(f'Individus doubles:\n')
    data_exp.print_all(dtfrm_duplications[index_name])

    # Affichage des indices des doublons par titre de film:
    group_doublon = dtfrm_duplications.groupby(index_name).apply
    (lambda x: list(x.index))

    print(f'Doublons et indices:\n')
    data_exp.print_all(group_doublon)

    return dtfrm_duplications, group_doublon


def diff_doublon(dtfrm: pd.DataFrame, attrbt, ftr_diffrncs):
    """Retourne les différences entre les deux dernières valeurs d'un
    feature pour des doublons d'une dataframe, et retourne l'identifiant
    du dernier doublon.

    Keyword arguments:
    attrbt -- attribut de la colonne sur laquelle on cherche les
    doublons (str)
    ftr_diffrncs -- feature de la colonne sur laquelle on calcule les
    différences (str)
    """

    # différences entre les deux dernières valeurs d'un attribut
    l_dif = []

    # identifiant du dernier doublon
    ident_last = []

    # Titres des films en double
    doublons_index = doublons(dtfrm, attrbt)[0][attrbt]

    for feature in doublons_index:
        # df des doublons pour chaque titre
        dfn = dtfrm.loc[dtfrm[attrbt] == feature]
        disp(dfn)

        # Différences entre les valeurs successives du feature pour des
        # doublons de l'attribut
        diffs = dfn[ftr_diffrncs].diff(periods=-1)
        # Différences entre les deux dernières valeurs
        l_dif.append(list(diffs)[-2])

        # identifiant du dernier doublon
        ident_last.append(diffs.index[-1])

    l_dif = trans(l_dif)
    ident_last = trans(ident_last)

    print(f'différences sur {ftr_diffrncs}:\n{diffs}')
    return l_dif, ident_last
