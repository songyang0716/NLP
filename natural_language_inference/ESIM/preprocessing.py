# Reference code https://github.com/coetaur0/ESIM
import sys
import numpy as np
import random
import re
# from string import maketrans
from string import punctuation

from collections import Counter

class preprocessor(object):
	"""
	Preprocessor class for natural language inference

	The class would be used to read NLI dataset, build worddicts
	and transfer the sentences to indexes
	"""
	def __init__(self, 
				 lowercase=True, 
				 ignore_punctuation=True, 
				 num_words=None, 
				 stopwords=[], 
				 labeldict={}, 
				 bos=None, 
				 eos=None):
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
		with open(filepath, 'r', encoding="utf-8") as f:
			ids, premises, hypotheses, labels = [], [], [], []
			# print(punctuation)
			punc_table = str.maketrans({punc:" " for punc in punctuation})
			# print(punc_table)
			next(f)
			for line in f:
				line = line.strip().split("\t")

				# Ignore sentences that have no gold label.
				if line[0] == "-":
					continue
					
				pair_id = line[8]
				premise = line[5]
				hypothesis = line[6]
				label = line[0]

				if self.lowercase:
					premise = premise.lower()
					hypothesis = hypothesis.lower()

				if self.ignore_punctuation:
					premise = premise.translate(punc_table)
					hypothesis = hypothesis.translate(punc_table)
				
				premises.append([word for word in premise.rstrip().split() if word not in self.stopwords])
				hypotheses.append([word for word in hypothesis.rstrip().split() if word not in self.stopwords])
				labels.append(label)
				ids.append(pair_id)

			return {"premises":premises,
					"hypotheses":hypotheses,
					"labels":labels,
					"ids":ids}


	def build_worddict(self, data):
		"""
		Build a dictionary associating words to unique integer indices for
		some dataset. The worddict can then be used to transform the words
		in datasets to their indices.
		Args:
			data: A dictionary containing the premises, hypotheses and
				labels of some NLI dataset, in the format returned by the
				'read_data' method of the Preprocessor class.
		"""
		
		
def main():
	processor = preprocessor()
	res = processor.read_data('./../data/snli_1.0/snli_1.0_train.txt')
	print(len(res['ids']))
	print(len(res['labels']))
	print(res['premises'][:10])
	print(res['hypotheses'][:10])


if __name__ == '__main__':
	main()










		
