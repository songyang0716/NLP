  
import collections
import os
import pickle
import random
import urllib
from io import open
import numpy as np
from collections import Counter
import nltk
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import en_core_web_sm 




def read_own_data(filename):
    """
    read your own data.
    :param filename:
    :return:
    """
    print('reading data...')
    with open(filename, 'r', encoding='utf-8') as f:
    	# data = f.read().splitlines()
    	data = f.readlines()
    print("corpus size", len(data))
    return data 

def build_dataset(reviews, n_words):
    """
    build dataset
    :param reviews: corpus
    :param n_words: learn most common n_words
    :return:
        - data: [word_index]
        - count: [ [word_index, word_count], ]
        - dictionary: {word_str: word_index}
        - reversed_dictionary: {word_index: word_str}
    """
    print("text cleaning...")
    nlp = en_core_web_sm.load()
    words = [word.text.lower() for review in reviews for word in nlp(review) if (not (word.is_stop) and (word.is_alpha))]
    print (len(words))
    print(words)
    print("text cleaning done...")
    return "", "", "", ""

