  
import collections
import os
import pickle
import random
import urllib
from io import open
import numpy as np



def read_own_data(filename):
    """
    read your own data.
    :param filename:
    :return:
    """
    print('reading data...')
    with open(filename, 'r', encoding='utf-8') as f:
    	data = f.read()
    print("corpus size", len(data))
    return data 
