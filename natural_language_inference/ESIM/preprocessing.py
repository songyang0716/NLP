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
			num_words: Numeber of frequent words we want to keep in the worddict
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

		self.n = len(ids)

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
		words = []
		for sentence in data["premises"]:
			for word in sentence:
				words.append(word)

		for sentence in data["hypotheses"]:
			for word in sentence:
				words.append(word)

		counts = Counter(words)

		num_words = self.num_words
		if self.num_words is None:
			num_words = len(counts)
		# print(num_words)
		# print(counts.most_common(10))
		
		self.worddict = {}
		self.idxword = {}

		# Special indices are used for padding, out-of-vocabulary words, and
		# beginning and end of sentence tokens.
		self.worddict["_PAD_"] = 0
		self.worddict["_OOV_"] = 1

		self.idxword[0] = "_PAD_"
		self.idxword[1] = "_OOV_"

		index = 2
		if self.bos:
			self.worddict["_BOS_"] = 2
			self.idxword[2] = "_BOS_"
			index += 1
		if self.eos:
			self.worddict["_EOS_"] = 3
			self.idxword[3] = "_EOS_"
			index += 1

		for word, freq in counts.most_common(num_words):
			self.worddict[word] = index
			self.idxword[index] = word
			index += 1

		# if self.labeldict == {}:
		label_names = set(data["labels"])
		self.labeldict = {label_name: i
						  for i, label_name in enumerate(label_names)}


	def words_to_indices(self, sentence):
		"""
		Transform the words in a sentence to their corresponding integer
		indices.
		Args:
			sentence: A list of words that must be transformed to indices.
		Returns:
			A list of indices.
		"""
		indices = []
		if self.bos:
			indices.append(2)
		for word in sentence:
			if word in self.worddict:
				indices.append(self.worddict[word])
			else:
				indices.append(1)
		if self.eos:
			indices.append(3)
		return indices
		

	def indices_to_words(self, indices):
		"""
		Transform the indices in a list to their corresponding words in
		the object's worddict.
		Args:
			indices: A list of integer indices corresponding to words in
				the Preprocessor's worddict.
		Returns:
			A list of words.
		"""
		words = []
		if self.bos:
			words.append("_BOS_")
		for index in indices:
			words.append(self.idxword[index])
		if self.eos:
			words.append("_EOS_")
		return words 


	def transform_to_indices(self, data):
		"""
		Transform the words in the premises and hypotheses of a dataset, as
		well as their associated labels, to integer indices.
		Args:
			data: A dictionary containing lists of premises, hypotheses
				and labels, in the format returned by the 'read_data'
				method of the Preprocessor class.
		Returns:
			A dictionary containing the transformed premises, hypotheses and
			labels.
		"""
		transformed_data = {"ids": [],
							"premises": [],
							"hypotheses": [],
							"labels": []}

		for i in range(self.n):
			transformed_data["ids"].append(data["ids"][i])

			transformed_data["labels"].append(self.labeldict[data["labels"][i]])

			premises_indices = self.words_to_indices(data["premises"][i])
			transformed_data["premises"].append(premises_indices)

			hypotheses_indices = self.words_to_indices(data["hypotheses"][i])
			transformed_data["hypotheses"].append(hypotheses_indices)

		return transformed_data


	def build_embedding_matrix(self, embeddings_file):
		"""
		Build an embedding matrix with pretrained weights for object's
		worddict.
		Args:
			embeddings_file: A file containing pretrained word embeddings.
		Returns:
			A numpy matrix of size (num_words+n_special_tokens, embedding_dim)
			containing pretrained word embeddings (the +n_special_tokens is for
			the padding and out-of-vocabulary tokens, as well as BOS and EOS if
			they're used).
		"""
		embeddings = {}
		with open(embeddings_file, 'r', encoding="utf-8") as f:
			for line in f:
				line = line.strip()
				line = line.split()
				word = line[0]
				embeddings[word] = line[1:]

		num_words = len(self.worddict)
		embedding_dim = len(list(embeddings.values())[0])
		embedding_matrix = np.zeros((num_words, embedding_dim))


		# Actual building of the embedding matrix.
		missed = 0
		for word, idx in self.worddict.items():
			if word in embeddings:
				embedding_matrix[idx] = np.array(embeddings[word], dtype=float)
			else:
				# PAD embedding will be all zeros
				if word == "_PAD_":
					continue
				missed += 1
				embedding_matrix[idx] = np.random.normal(size=(embedding_dim))
				# print(word)
		print("Missed words: ", missed)
		print("Total missed words in the glove embedding over all words: ", missed / num_words)
		return embedding_matrix



class NLIDataset(Dataset):
    """
    Dataset class for Natural Language Inference datasets.
    The class can be used to read preprocessed datasets where the premises,
    hypotheses and labels have been transformed to unique integer indices
    (this can be done with the 'preprocess_data' script in the 'scripts'
    folder of this repository).
    """
    

    


def main():
	snli_text_path = "./../data/snli_1.0/snli_1.0_train.txt"
	embedding_file = "./../../data/glove.6B/glove.6B.50d.txt"

	processor = preprocessor(num_words=1000)
	res = processor.read_data(snli_text_path)
	# print(res['labels'][:10])
	# print(res['premises'][:10])
	# print(res['hypotheses'][:10])
	# print(res['ids'][:10])
	processor.build_worddict(res)
	res_2 = processor.transform_to_indices(res)
	embedding_matrix = processor.build_embedding_matrix(embedding_file)
	# print(res_2['labels'][:10])
	# print(res_2['premises'][:10])
	# print(res_2['hypotheses'][:10])
	# print(res_2['ids'][:10])



if __name__ == '__main__':
	main()










		
