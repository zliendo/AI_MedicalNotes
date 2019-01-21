###This file contains the functions reextracting the database only for the top ICD codes

import numpy as np
import pandas as pd
from collections import Counter


def find_top_codes(df, col_name, n):
    """ Find the top codes from a columns of strings
        Returns a list of strings to make sure codes are treated as classes down the line """
    string_total = df[col_name].str.cat(sep=' ')
    counter_total = Counter(string_total.split(' '))
    return [word for word, word_count in counter_total.most_common(n)]

def select_codes_in_string(string, top_codes):
    """ Creates a sring of the codes which are both in the original string
        and in the top codes list """
    r = ''
    for code in top_codes:
        if code in string:
            r += ' ' + code
    return r.strip()

def filter_top_codes(df, col_name, n, filter_empty = True):
    """ Creates a dataframe with the codes column containing only the top codes
        and filters out the lines without any of the top codes if True
        
        Note: we may actually want to keep even the empty lines """
    r = df.copy()
    top_codes = find_top_codes(r, col_name, n)
    r[col_name] = r[col_name].apply(lambda x: select_codes_in_string(x, top_codes))
    if filter_empty:
        r = r.loc[r[col_name] != '']
    return r, top_codes
