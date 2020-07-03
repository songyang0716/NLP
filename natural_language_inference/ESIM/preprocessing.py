# Reference code https://github.com/coetaur0/ESIM
import sys
import numpy as np
import random
import re

from collections import Counter

class preprocessor(object):
	"""
	Preprocessor class for natural language inference

	The class would be used to read NLI dataset, build worddicts
	and transfer the sentences to indexes
	"""
	def __init__(self,
				 lowercase=True,
				 ignore_punctuation=False,
				 num_words=None,
				 stopwords=[],
				 labeldict={},
				 bos=None,
				 eos=None)
	"""
	Args:
		lowercase: A booleana to indicate whether the words to be lowercased or not
		ignore_punctuation: A boolean to indicate whether ignore punctuation
		stopwords: A list of words that would be ignored when building worddict
		num_words: Numeber of words we want to keep in the worddict
		bos: Indicate the symbol to use for begining of sentence
		eos: Indicate the symbol to use for ending of sentence
	"""
	self.lowercase = lowercase
	self.ignore_punctuation = ignore_punctuation
	self.num_words = num_words
	self.stopwords = stopwords
	self.labeldict = labeldict
	self.bos = bos 
	self.eos = eos 

	def read_data(self, filepath):
		"""
		Read the premises, hypotheses and labels from some NLI dataset's
		file and return them in a dictionary. The file should be in the same
		form as SNLI's .txt files.
		Args:
			filepath: The path to a file containing some premises, hypotheses
				and labels that must be read. The file should be formatted in
				the same way as the SNLI (and MultiNLI) dataset.
		Returns:
			A dictionary containing three lists, one for the premises, one for
			the hypotheses, and one for the labels in the input data.
		"""
		


















		
