# Reference from https://github.com/albertwy/BiLSTM/blob/master/main_batch.py

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from model import BLSTM 



batch_size = 80
embsize = 50
hidden_size = 64
n_layers = 1
max_len = 40
dropout = 0
l_rate = 0.01
input_dir = "./data/"



def get_padding(sentences, max_len):
	"""
	:param sentences: raw sentence --> index_padded sentence
					[2, 3, 4], 5 --> [2, 3, 4, 0, 0]
	:param max_len: number of steps to unroll for a LSTM
	:return: sentence of max_len size with zero paddings , and also the real lenght of each sentence (capped under max-len)
	"""
	seq_len = np.zeros((0,))
	padded = np.zeros((0, max_len))
	for sentence in sentences:
		num_words = len(sentence)
		num_pad = max_len - num_words
		sentence = np.asarray(sentence[:max_len], dtype=np.int64).reshape(1, -1)
		if num_pad > 0:
			zero_paddings = np.zeros((1, num_pad), dtype=np.int64)
			sentence = np.concatenate((sentence, zero_paddings), axis=1)
		else:
			num_words = max_len
		padded = np.concatenate((padded, sentence), axis=0)
		# seq_len will be used in the BiLSTM model, to find the last hidden layer 
		seq_len = np.append(seq_len, num_words)
	return padded.astype(np.int64), seq_len.astype(np.int64)





class YDataset(object):
	def __init__(self, features, labels, to_pad=True, max_len=40):
		"""
		:param features: list containing sequences to be padded and batched, all sequences are indexes of words
		:param labels:
		"""
		self.features = features
		self.labels = labels
		self.pad_max_len = max_len
		self.seq_lens = None
		# self.mask_matrix = None

		assert len(features) == len(self.labels)

		self._num_examples = len(self.labels)
		self._epochs_completed = 0
		self._index_in_epoch = 0

		if to_pad:
			if max_len:
				self._padding()
				# self._mask()
			else:
				print("Need more information about padding max_length")

	def __len__(self):
		return self._num_examples


	def _padding(self):
		self.features, self.seq_lens = get_padding(self.features, max_len=self.pad_max_len)
		print(self.seq_lens)
		print(len(self.seq_lens))
		# print(self.features)


	def _shuffle(self, seed):
		"""
		After each epoch, the data need to be shuffled
		:return:
		"""
		perm = np.arange(self._num_examples)
		np.random.shuffle(perm)

		self.features = self.features[perm]
		self.seq_lens = self.seq_lens[perm]
		# self.mask_matrix = self.mask_matrix[perm]
		self.labels = self.labels[perm]

	def next_batch(self, batch_size, seed=888):
		"""Return the next `batch_size` examples from this data set."""
		start = self._index_in_epoch
		self._index_in_epoch += batch_size
		if self._index_in_epoch > self._num_examples:
			# Finished epoch
			self._epochs_completed += 1
			'''  shuffle feature and labels'''
			self._shuffle(seed=seed)
			start = 0
			self._index_in_epoch = batch_size
			assert batch_size <= self._num_examples

		end = self._index_in_epoch

		features = self.features[start:end]
		seq_lens = self.seq_lens[start:end]
		# mask_matrix = self.mask_matrix[start:end]
		labels = self.labels[start:end]

		return features, seq_lens, labels
		

##############################
#   Serialization to pickle  #
##############################
def pickle2dict(in_file):
	try:
		import cPickle as pickle
	except ImportError:
		import pickle
	with open(in_file, 'rb') as f:
		your_dict = pickle.load(f)
		return your_dict


def main():
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(device)

	dataset = pickle2dict(input_dir + "features_glove.pkl")
	embeddings = pickle2dict(input_dir + "embeddings_glove.pkl")
	dataset["embeddings"] = embeddings

	emb_np = np.asarray(embeddings, dtype=np.float32)
	emb = torch.from_numpy(emb_np)


	blstm_models = BLSTM(embeddings=emb,
						input_dim=embsize,
						hidden_dim=hidden_size,
						num_layers=n_layers,
						output_dim=2,
						max_len=max_len,
						dropout=dropout)

	blstm_models = blstm_models.to(device)

	optimizer = optim.SGD(blstm_models.parameters(), lr=l_rate, weight_decay=1e-5)
	criterion = nn.CrossEntropyLoss()

	training_set = dataset["training"]
	training_set = YDataset(training_set["xIndexes"],
							training_set["yLabels"],
							to_pad=True,
							max_len=max_len)

	best_acc_test, best_acc_valid = -np.inf, -np.inf
	batches_per_epoch = int(len(training_set)/batch_size)
	# print(batches_per_epoch)
	# print(training_set)

	# print(len(training_set['xIndexes']))
	# print(len(training_set['yLabels']))
	# print(blstm_models)


if __name__ == '__main__':
	main()






























