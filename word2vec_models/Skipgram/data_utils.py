###############################################################################################################
#### Reference: https://github.com/blackredscarf/pytorch-SkipGram/blob/master/word2vec.py
################################################################################################################ 
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
		data = f.readlines()
	# print("corpus size", len(data))
	return data 

def build_dataset(reviews, n_words):
	"""
	build dataset
	:param reviews: corpus
	:param n_words: learn most common n_words
	:return:
		- words_ix: [word_index]
		- words_freq_dict: {word_index: word_count}
		- word_to_ix: {word_str: word_index}
		- ix_to_word: {word_index: word_str}
	"""
	print("text cleaning...")
	nlp = en_core_web_sm.load()
	words = [word.text.lower() for review in reviews for word in nlp(review) if (not (word.is_stop) and (word.is_alpha))]
	words_freq = collections.Counter(words)
	words_freq = sorted(words_freq.items(), key=lambda x: x[1], reverse=True)

	words_freq_dict = collections.defaultdict(int)

	for word, freq in words_freq[:n_words-1]:
		words_freq_dict[word] = freq

	for _, freq in words_freq[n_words-1:]:
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

def noise(vocabs, word_count, word2index, index2word, use_noise_neg=True):
	"""
	generate noise distribution
	:param vocabs:
	:param word_count:
	:return:
	"""
	# unigram_table provide the probability of the word being drawn, the index is the word index
	unigram_table = []
	num_total_words = sum([c for w,c in word_count.items()])
	for vocab in vocabs:
		if not use_noise_neg:
			unigram_table.append(word_count[index2word[vocab]] / num_total_words)
		else:
			unigram_table.append((word_count[index2word[vocab]] / num_total_words)**(3/4))
	unigram_table = [i/sum(unigram_table) for i in unigram_table]
	return unigram_table



class DataPipeline:
	def __init__(self, data, vocabs, word_count, word2index, index2word):
		"""
		: data:  words_ix [word_index]
		: vocabs: set of unique words (unigrams)
		: word_count: {word_index: word_count}
		"""
		self.data = data
		self.unigram_table = noise(vocabs, word_count, word2index, index2word, use_noise_neg=True)
		# print(self.unigram_table)


	
	def get_neg_data(self, batch_size, num):
		"""
		sample the negative data. Don't use np.random.choice(), it is very slow.
		:param batch_size: int
		:param num: int
		:return:
		"""
		neg = np.zeros(shape=(batch_size, num),dtype=int)
		for i in range(batch_size):
			neg_for_cur_batch = np.random.choice(a=len(self.unigram_table), size=num, replace=False, p=self.unigram_table)
			neg[i] = neg_for_cur_batch
		# print(neg)
		# print(neg.shape)
		return neg




	def generate_batch(self, batch_size, num_skips, cur_batch_index, n_words):
		"""
		get the data batch
		:param batch_size:
		:param num_skips:
		:return: target batch and context batch
		""" 
		batch = np.zeros(shape=batch_size,dtype=int)
		context = np.zeros(shape=batch_size,dtype=int)
		start_index = cur_batch_index*batch_size
		for i in range(batch_size):
			batch[i] = self.data[cur_batch_index*batch_size + i]
			window_left_margin =  max(0, cur_batch_index*batch_size + i - num_skips)
			window_right_margin = min(cur_batch_index*batch_size + i + num_skips + 1, n_words)
			candidates = [j for j in range(window_left_margin, window_right_margin) if j != (cur_batch_index*batch_size + i)]
			context[i] = self.data[random.choice(candidates)]	
			# print("Cur word")
			# print(cur_batch_index*batch_size + i)
			# print("Left margin")		
			# print(window_left_margin)
			# print("Right margin")
			# print(window_right_margin)
			# print("Selected context")
			# print(context[i])
		return batch, context





