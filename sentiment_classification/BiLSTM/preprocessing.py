"""
Preprocess the movie reviews sentences
"""

import sys
import numpy as np
import random
import re



# parse raw text into list (sentence in strings)
def parse_file2list(filename):
	contents = []
	with open(filename, 'r', encoding = "ISO-8859-1") as f:
		contents = [line.split("\n")[0] for line in f]
	print ("The file has lines: {}".format(len(contents)))
	return contents


# parse segmented corpus into list (sentence in list of words)
def parse_file2lol(filename):
	with open(filename, 'r', encoding = "ISO-8859-1") as f:
		contents = [string2list(line) for line in f]
	print ("The file has lines: {}".format(len(contents)))
	return contents


def string2list(sentence_in_string):
	"""convert strings with '\n' to list of words without '\n' """
	return sentence_in_string.strip().replace('-',' ').split()


def shuffle(lol, seed=888):
	"""
	lol : list of list as input, in our example, lol is a list of sentences & labels 
		  lol[0] is sentences, lol[1] is labels
	seed : seed the shuffling
	
	shuffle inplace each list in the same order
	"""
	for l in lol:
		random.seed(seed)
		random.shuffle(l)


def read_emb_idx(filename):
	"""
	1.read embeddings files to
		"embeddings": numpy matrix, each row is a vector with corresponding index
		"word2idx": word2idx[word] = idx in the "embeddings" matrix
		"idx2word": the reverse dict of "word2idx"
	2. add padding and unk to 3 dictionaries
	:param filename:
		file format: word<space>emb, '\n' (line[0], line[1:-1], line[-1])
	:return:
		vocab = {"embeddings": embeddings, "word2idx": word2idx, "idx2word": idx2word}
	"""
	with open(filename, 'r', encoding="utf-8") as f:
		embeddings = []
		word2idx = dict()

		word2idx["_padding"] = 0  # PyTorch Embedding lookup need padding to be zero
		word2idx["_unk"] = 1

		for line in f:
			line = line.strip()
			line = line.split()
			word = line[0]
			emb = [float(v) for v in line[1:]]
			embeddings.append(emb)
			word2idx[word] = len(word2idx)
			
		''' Add padding and unknown word to embeddings and word2idx'''
		emb_dim = len(embeddings[0])
		embeddings.insert(0, np.zeros(emb_dim))  # _padding
		embeddings.insert(1, np.random.random(emb_dim))  # _unk

		embeddings = np.asarray(embeddings, dtype=np.float32)
		embeddings = embeddings.reshape(len(embeddings), emb_dim)

		idx2word = dict((word2idx[word], word) for word in word2idx)
		
		vocab = {"embeddings": embeddings, "word2idx": word2idx, "idx2word": idx2word}

		print ("Finish loading embedding {} * * * * * * * * * * * *".format(filename))
		return vocab


def sentence_to_index(word2idx, sentences):
	"""
	Transform sentence into lists of word index
	:param word2idx:
		word2idx = {word:idx, ...}
	:param sentences:
		list of sentences which are list of word
	:return:
	"""
	print ("-------------begin making sentence xIndexes-------------")
	sentences_indexes = []
	for sentence in sentences:
		s_index = []
		for word in sentence:
			if word in word2idx:
				s_index.append(word2idx[word])
			else:
				s_index.append(word2idx["_unk"])
				# print("Can't find the word {} in the embedding list".format(word))
		if not s_index:
			# print("Empty sentences: {}".format(sentence))
			s_index.append(word2idx["_unk"])
		sentences_indexes.append(s_index)

	assert len(sentences_indexes) == len(sentences)
	print ("-------------finish making sentence xIndexes-------------")
	return sentences_indexes

def make_datasets(word2idx, raw_data):
	"""
	:param word2idx:
		word2idx = {word:idx, ...}
	:param raw_data:
		raw_data = {"training": (inputs, labels),
					"validation",
					"test"}
	:return:
	"""
	datasets = dict()

	for i in ["training", "validation", "test"]:
		sentences, labels = raw_data[i]
		xIndexes = sentence_to_index(word2idx, sentences)
		yLabels = [int(label) for label in labels]
		yLabels = np.asarray(yLabels, dtype=np.int64).reshape(len(labels))
		datasets[i] = {"xIndexes": xIndexes,
					   "yLabels": yLabels}

	return datasets


###################
#   Serialization to pickle
###################
def dict2pickle(your_dict, out_file):
	try:
		import cPickle as pickle
	except ImportError:
		import pickle
	with open(out_file, 'wb') as f:
		pickle.dump(your_dict, f)




def processing():
	input_dir = "./data/"
	output_dir = input_dir
	embedding_file = "./../../data/glove.6B/glove.6B.50d.txt"
	# read sentences & labels
	data = [] 
	files = ["MR.task.train",
			 "MR.task.test"]
	for file in files:
		# sentences, labels
		sentences = parse_file2lol(input_dir + file + ".sentences")
		labels = parse_file2list(input_dir + file + ".labels")
		data.append([sentences, labels])
	
	assert len(data[0][0]) == len(data[0][1])
	assert len(data[1][0]) == len(data[1][1])
	
	# split the dataset: train, test
	shuffle(data[0], seed=888)
	train = data[0]
	test = data[1]
	# assume we don't have validation set, just use test
	valid = test

	assert len(train[0]) == len(train[1])
	assert len(valid[0]) == len(valid[1])
	assert len(test[0])  == len(test[1])

	raw_data = {"training": train,
				"validation": valid,
				"test": test}

	vocab = read_emb_idx(embedding_file)
	word2idx, embeddings = vocab["word2idx"], vocab["embeddings"]

	# transform sentence to word index
	datasets = make_datasets(word2idx, raw_data)

	dict2pickle(datasets, output_dir + "features_glove.pkl")
	dict2pickle(word2idx, output_dir + "word2idx_glove.pkl")
	dict2pickle(embeddings, output_dir + "embeddings_glove.pkl")


	print (word2idx["_padding"])
	print (word2idx["_unk"])
	print ("-------------Finish processing-------------")


if __name__ == '__main__':
	processing()

