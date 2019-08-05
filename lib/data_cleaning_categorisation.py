# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 15:21:50 2018

@author: Mahery
"""
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
