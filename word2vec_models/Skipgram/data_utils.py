  
import collections
import os
import pickle
import random
import urllib
from io import open
import numpy as np
from collections import Counter
import nltk
################################################################################################################
#### Reference: https://github.com/blackredscarf/pytorch-SkipGram/blob/master/word2vec.py
################################################################################################################
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
	words_freq = collections.Counter(words)
	words_freq = sorted(words_freq.items(), key=lambda x: x[1], reverse=True)

	words_freq_dict = collections.defaultdict(int)

	for word, freq in words_freq[:n_words]:
		words_freq_dict[word] = freq

	for _, freq in words_freq[n_words:]:
		words_freq_dict['UNK'] += freq

	vocab = words_freq_dict.keys()
	vocab_size = len(vocab)
	
	word_to_ix = {word: i for i, word in enumerate(vocab)}
	ix_to_word = {i: word for i, word in enumerate(vocab)}

	words_ix = []

	for word in words:
		if word in words_freq_dict:
			words_ix.append(word_to_ix[word])
		else:
			words_ix.append(word_to_ix['UNK'])

	print("text cleaning done...")
	return words_ix, words_freq_dict, word_to_ix, ix_to_word



class DataPipeline:
    def __init__(self, data, vocabs, word_count, data_index=0, use_noise_neg=True):
	
    def get_neg_data(self, batch_size, num, target_inputs):
        """
        sample the negative data. Don't use np.random.choice(), it is very slow.
        :param batch_size: int
        :param num: int
        :param target_inputs: []
        :return:
        """
    
    def generate_batch(self, batch_size, num_skips, skip_window):
        """
        get the data batch
        :param batch_size:
        :param num_skips:
        :param skip_window:
        :return: target batch and context batch
        """ 






