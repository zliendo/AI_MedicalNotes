### This file contains the functions necessary to vectorize the ICD labels and text inputs

import numpy as np
import pandas as pd
import re
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# Vectorize ICD codes

def vectorize_icd_string(x, code_list):
    """Takes a string with ICD codes and returns an array of the right of 0/1"""
    r = []
    for code in code_list:
        if code in x: r.append(1)
        else: r.append(0)
    return np.asarray(r)

def vectorize_icd_column(df, col_name, code_list):
    """Takes a column and applies the """
    r = df[col_name].apply(lambda x: vectorize_icd_string(x, code_list))
    r = np.transpose(np.column_stack(r))
    return r


# Clean Text

def clean_str(string):
    """Cleaning of notes"""

    """ Cleaning from Guillaume """
    string = string.lower()
    string = string.replace("\n", " ") # remove the lines
    string = re.sub("\[\*\*.*?\*\*\]", "", string) # remove the things inside the [** **]
    string = re.sub("[^a-zA-Z0-9\ \']+", " ", string)

    """ Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
    #string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    #string = re.sub(r",", " , ", string)
    #string = re.sub(r"!", " ! ", string)
    #string = re.sub(r"\(", " \( ", string)
    #string = re.sub(r"\)", " \) ", string)
    #string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    
    """ Canonize numbers"""
    string = re.sub(r"(\d+)", "DG", string)
    
    return string.strip()

def clean_notes(df, col_name):
    r = df[col_name].apply(lambda x: clean_str(x))
    return r


# Vectorize and Pad notes Text

def vectorize_notes(col, MAX_NB_WORDS, verbose = True):
    """Takes a note column and encodes it into a series of integer
        Also returns the dictionnary mapping the word to the integer"""
    tokenizer = Tokenizer(num_words = MAX_NB_WORDS)
    tokenizer.fit_on_texts(col)
    data = tokenizer.texts_to_sequences(col)
    note_length =  [len(x) for x in data]
    vocab = tokenizer.word_index
    MAX_VOCAB = len(vocab)
    if verbose:
        print('Vocabulary size: %s' % MAX_VOCAB)
        print('Average note length: %s' % np.mean(note_length))
        print('Max note length: %s' % np.max(note_length))
    return data, vocab, MAX_VOCAB

def pad_notes(data, MAX_SEQ_LENGTH):
    data = pad_sequences(data, maxlen = MAX_SEQ_LENGTH)
    return data, data.shape[1]


# Creates an embedding Matrix
# Based on https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html

def embedding_matrix(f_name, dictionary, EMBEDDING_DIM, verbose = True, sigma = None):
    """Takes a pre-trained embedding and adapts it to the dictionary at hand
        Words not found will be all-zeros in the matrix"""

    # Dictionary of words from the pre trained embedding
    pretrained_dict = {}
    with open(f_name, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            pretrained_dict[word] = coefs

    # Default values for absent words
    if sigma:
        pretrained_matrix = sigma * np.random.rand(len(dictionary) + 1, EMBEDDING_DIM)
    else:
        pretrained_matrix = np.zeros((len(dictionary) + 1, EMBEDDING_DIM))
    
    # Substitution of default values by pretrained values when applicable
    for word, i in dictionary.items():
        vector = pretrained_dict.get(word)
        if vector is not None:
            pretrained_matrix[i] = vector

    if verbose:
        print('Vocabulary in notes:', len(dictionary))
        print('Vocabulary in original embedding:', len(pretrained_dict))
        inter = list( set(dictionary.keys()) & set(pretrained_dict.keys()) )
        print('Vocabulary intersection:', len(inter))

    return pretrained_matrix, pretrained_dict
