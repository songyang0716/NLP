################################################################################################################
#### Reference: https://github.com/blackredscarf/pytorch-SkipGram/blob/master/word2vec.py
################################################################################################################
from data_utils import read_own_data, build_dataset
import sys
from model import SkipGramNeg
import torch.optim as optim


class Word2Vec:
	def __init__(self, path, vocab_size=50, n_review=500, embedding_size=100, learning_rate=1.0):
		self.corpus = read_own_data(path)
		self.corpus = self.corpus[:n_review]
		self.data, self.word_count, self.word2index, self.index2word = build_dataset(self.corpus, vocab_size)
		# print(self.word_count)
		# print(self.word2index)
		# print(self.index2word)
		# print(self.data)
		self.vocabs = list(set(self.data))
		self.model = SkipGramNeg(vocab_size, embedding_size).cuda()
		self.model_optim = optim.SGD(self.model.parameters(), lr=learning_rate)


	def train(self):
		print("abc")







review_path = '/Users/yangsong/Desktop/Projects/gitrepo_songyang0716/NLP/word2vec_models/yelp_review_10000.txt'
w2v = Word2Vec(review_path, vocab_size=100)
# w2v.train()
# print(w2v.corpus[0])
