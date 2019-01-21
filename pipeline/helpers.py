### This contains helper functions

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score



def train_val_test_split(X, y, val_size=0.2, test_size=0.2, random_state=101):
    """Splits the input and labels into 3 sets"""
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(val_size+test_size), random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size/(val_size+test_size), random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test


def replace_with_grandparent_codes(string_codes, ICD9_FIRST_LEVEL):
    """replace_with_grandparent_codes takes a list of ICD9 codes and 
    returns the list of their grandparents ICD9 code in the first level of the ICD9 hierarchy"""
    ICD9_RANGES = [x.split('-') for x in ICD9_FIRST_LEVEL]
    resulting_codes = []
    for code in string_codes.split(' '):
        for i,gparent_range in enumerate(ICD9_RANGES):
            range = gparent_range[1] if len(gparent_range) == 2 else gparent_range[0]
            if code[0:3] <= range:
                resulting_codes.append(ICD9_FIRST_LEVEL[i])
                break 
    return ' '.join (set(resulting_codes))


def write_col(df, col_name, fname):
    df[col_name].to_csv(fname)
    

def get_f1_score(y_true,y_hat,threshold, average):
    hot_y = np.where(np.array(y_hat) > threshold, 1, 0)
    return f1_score(np.array(y_true), hot_y, average=average)

def show_f1_score(y_train, pred_train, y_val, pred_dev):
    print('F1 scores')
    print('threshold | training | dev  ')
    f1_score_average = 'micro'
    for threshold in [ 0.02, 0.03,0.04,0.05,0.055,0.058,0.06, 0.08, 0.1,0.2,0.3, 0.4, 0.5, 0.6,0.7]:
        train_f1 = get_f1_score(y_train, pred_train,threshold,f1_score_average)
        dev_f1 = get_f1_score(y_val, pred_dev,threshold,f1_score_average)
        print('%1.3f:      %1.3f      %1.3f' % (threshold,train_f1, dev_f1))