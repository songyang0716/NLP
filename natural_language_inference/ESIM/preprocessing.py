# Reference code: https://github.com/coetaur0/ESIM
import sys
import numpy as np
import random
import re
import os
import pickle
# from string import maketrans
from string import punctuation
from torch.utils.data import Dataset
from collections import Counter
import torch 

class Preprocessor(object):
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


# Custom dataset. inherit Dataset and override the following methods:
# __len__ so that len(dataset) returns the size of the dataset.
# __getitem__ to support the indexing such that dataset[i] can be used to get ith sample
class NLIDataset(Dataset):
	"""
	Dataset class for Natural Language Inference datasets.
	The class can be used to read preprocessed datasets where the premises,
	hypotheses and labels have been transformed to integer indices
	"""
	def __init__(self,
				 data,
				 padding_idx=0,
				 max_premise_length=None,
				 max_hypothesis_length=None):
		"""
		Args:
			data: A dictionary containing the preprocessed premises,
				hypotheses and labels of some dataset.
			padding_idx: An integer indicating the index being used for the
				padding token in the preprocessed data. Defaults to 0.
			max_premise_length: An integer indicating the maximum length
				accepted for the sequences in the premises. If set to None,
				the length of the longest premise in 'data' is used.
				Defaults to None.
			max_hypothesis_length: An integer indicating the maximum length
				accepted for the sequences in the hypotheses. If set to None,
				the length of the longest hypothesis in 'data' is used.
				Defaults to None.
		"""
		self.premises_lengths = [len(seq) for seq in data["premises"]]
		self.max_premise_length = max_premise_length
		if self.max_premise_length is None:
			self.max_premise_length = max(self.premises_lengths)


		self.hypotheses_lengths = [len(seq) for seq in data["hypotheses"]]
		self.max_hypothesis_length = max_hypothesis_length
		if self.max_hypothesis_length is None:
			self.max_hypothesis_length = max(self.hypotheses_lengths)
		
		# print(self.max_premise_length)
		# print(self.max_hypothesis_length)
		self.num_sequences = len(data["premises"])

		self.data = {"ids": [],
					 "premises": torch.zeros((self.num_sequences,
											  self.max_premise_length),
											  dtype=torch.long),
					 "hypotheses": torch.zeros((self.num_sequences,
												self.max_hypothesis_length),
												dtype=torch.long),
					 "labels": torch.tensor(data["labels"], 
											dtype=torch.long)}

		for i in range(self.num_sequences):
			self.data["ids"].append(data["ids"][i])

			end = min(len(data["premises"][i]), self.max_premise_length)
			self.data["premises"][i][:end] = torch.tensor(data["premises"][i][:end])

			end = min(len(data["hypotheses"][i]), self.max_premise_length)
			self.data["hypotheses"][i][:end] = torch.tensor(data["hypotheses"][i][:end])

	def __getitem__(self, index):
		return {"id": self.data["ids"][index],
				"premise": self.data["premises"][index],
				"premise_length": min(self.premises_lengths[index],
									  self.max_premise_length),
				"hypothesis": self.data["hypotheses"][index],
				"hypothesis_length": min(self.hypotheses_lengths[index],
										 self.max_hypothesis_length),
				"label": self.data["labels"][index]}


	def __len__(self):
		return self.num_sequences


# def dict2pickle(your_dict, out_file):
# 	try:
# 		import cPickle as pickle
# 	except ImportError:
# 		import pickle
# 	with open(out_file, 'wb') as f:
# 		pickle.dump(your_dict, f)


def main():
	snli_text_train_path = "./../data/snli_1.0/snli_1.0_train.txt"
	snli_text_dev_path = "./../data/snli_1.0/snli_1.0_dev.txt"
	snli_text_test_path = "./../data/snli_1.0/snli_1.0_test.txt"

	embedding_file = "./../../data/glove.6B/glove.6B.50d.txt"
	output_dir = "./../data/"

	# -------------------- Train data preprocessing -------------------- #
	preprocessor = Preprocessor(num_words=10000)
	data = preprocessor.read_data(snli_text_train_path)

	preprocessor.build_worddict(data)
	# res_2 = processor.transform_to_indices(res)
	# embedding_matrix = processor.build_embedding_matrix(embedding_file)
	
	with open(os.path.join(output_dir, "worddict.pkl"), "wb") as pkl_file:
		pickle.dump(preprocessor.worddict, pkl_file)

	print("\t* Transforming words in premises and hypotheses to indices...")
	transformed_data = preprocessor.transform_to_indices(data)
	print("\t* Saving result...")
	with open(os.path.join(output_dir, "train_data.pkl"), "wb") as pkl_file:
		pickle.dump(transformed_data, pkl_file)


	# -------------------- Validation data preprocessing -------------------- #
	print(20*"=", " Preprocessing dev set ", 20*"=")
	print("\t* Reading data...")
	data = preprocessor.read_data(snli_text_dev_path)

	print("\t* Transforming words in premises and hypotheses to indices...")
	transformed_data = preprocessor.transform_to_indices(data)
	print("\t* Saving result...")
	with open(os.path.join(output_dir, "dev_data.pkl"), "wb") as pkl_file:
		pickle.dump(transformed_data, pkl_file)

	# -------------------- Test data preprocessing -------------------- #
	print(20*"=", " Preprocessing test set ", 20*"=")
	print("\t* Reading data...")
	data = preprocessor.read_data(snli_text_test_path)

	print("\t* Transforming words in premises and hypotheses to indices...")
	transformed_data = preprocessor.transform_to_indices(data)
	print("\t* Saving result...")
	with open(os.path.join(output_dir, "test_data.pkl"), "wb") as pkl_file:
		pickle.dump(transformed_data, pkl_file)



	# -------------------- Embeddings preprocessing -------------------- #
	print(20*"=", " Preprocessing embeddings ", 20*"=")
	print("\t* Building embedding matrix and saving it...")
	embed_matrix = preprocessor.build_embedding_matrix(embedding_file)
	with open(os.path.join(output_dir, "embeddings.pkl"), "wb") as pkl_file:
		pickle.dump(embed_matrix, pkl_file)



if __name__ == '__main__':
	main()
