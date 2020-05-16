###############################################################################################################
### Reference: https://github.com/blackredscarf/pytorch-SkipGram/blob/master/word2vec.py        ###############
###############################################################################################################
from data_utils import read_own_data, build_dataset, DataPipeline
import sys
from model import SkipGramNeg
import torch.optim as optim
import random

class Word2Vec:
	def __init__(self, path, vocab_size=50, n_review=10, embedding_size=100, learning_rate=1.0):
		self.corpus = read_own_data(path)
		self.corpus = self.corpus[:n_review]
		self.data, self.word_count, self.word2index, self.index2word = build_dataset(self.corpus, vocab_size)
		self.vocabs = list(set(self.data))
		# self.model = SkipGramNeg(vocab_size, embedding_size).cuda()
		self.model = SkipGramNeg(vocab_size, embedding_size)
		self.model_optim = optim.SGD(self.model.parameters(), lr=learning_rate)

		print("Number of token is: ", len(self.data))
		# print(self.vocabs)
		# print(len(self.vocabs))  51
		# print(self.vocabs)       index [0,1,2,3....]
		# print(len(self.word_count))  
		# print(self.word_count)   default dict, word:count
		# print(self.data)		   original text, use index	



	def train(self, 
			  train_steps=10, 
			  skip_window=1, 
			  num_skips=2, 
			  num_neg=20, 
			  batch_size=128, 
			  data_offest=0, 
			  vali_size=3, 
			  output_dir='out'):
		# self.outputdir = os.mkdir(output_dir)
		avg_loss = 0
		pipeline = DataPipeline(self.data, self.vocabs ,self.word_count, self.word2index, self.index2word, data_offest)
		# used during the training analysis as validation samples
		vali_example = random.sample(self.vocabs, vali_size)
		# print(vali_example)
		# [3, 20, 50]




review_path = '/Users/yangsong/Desktop/Projects/gitrepo_songyang0716/NLP/word2vec_models/yelp_review_10000.txt'
w2v = Word2Vec(review_path)
w2v.train()
# print(w2v.corpus[0])







