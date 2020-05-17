from data_utils import read_own_data, build_dataset, DataPipeline
import sys
from model import SkipGramNeg
import torch
import torch.optim as optim
import random
import numpy as np


class Word2Vec:
	def __init__(self, path, vocab_size=100, n_review=200, embedding_size=50, learning_rate=0.05):
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
			  epochs=1, 
			  num_skips=2, 
			  num_neg=5, 
			  batch_size=32, 
			  vali_size=3, 
			  output_dir='out'):
		# self.outputdir = os.mkdir(output_dir)
		pipeline = DataPipeline(self.data, self.vocabs ,self.word_count, self.word2index, self.index2word)
		# used during the training analysis as validation samples
		vali_example = random.sample(self.vocabs, vali_size)
		# print(vali_example)
		# [3, 20, 50]
		n_batches = len(self.data) // batch_size
		n_words = len(self.data)

		for epoch in range(epochs):
			avg_loss = 0
			for batch in range(n_batches):
				batch_inputs, batch_labels = pipeline.generate_batch(batch_size, num_skips, batch, n_words)
				batch_neg = pipeline.get_neg_data(batch_size, num_neg)

				batch_inputs = torch.tensor(batch_inputs, dtype=torch.long)
				batch_labels = torch.tensor(batch_labels, dtype=torch.long)
				batch_neg = torch.tensor(batch_neg, dtype=torch.long)

				# print("inputs")
				# print(batch_inputs)
				# print("labels")
				# print(batch_labels)
				# print("negative")
				# print(batch_neg)

				loss = self.model(batch_inputs, batch_labels, batch_neg)
				self.model_optim.zero_grad()
				loss.backward()
				self.model_optim.step()

				avg_loss += loss
			avg_loss = avg_loss/n_batches

			print('Total loss at current epoch : ', avg_loss)

			# if epoch % 100 == 0 and vali_size > 0:
			if epoch == 0:
				print(vali_example)
				self.most_similar(vali_example)



	def most_similar(self, word_idx, top_k=5):
		index = torch.tensor(word_idx, dtype=torch.long)
		emb = self.model.predict(index).squeeze(0)
		sim = torch.mm(emb, self.model.input_emb.weight.transpose(0, 1))
		print(sim.shape)
		print(sim)
		sorted_index = torch.argsort(sim, dim=1, descending=True)
		print(sorted_index.shape)
		# print(emb)
		# print(emb.squeeze(0).shape)
		# print(self.model.input_emb.weight.shape)
		# sim = torch.mm(emb, self.model.input_emb.weight.transpose(0, 1))
		# nearest = (-sim[0]).sort()[1][1: top_k + 1]
		# top_list = []
		# for k in range(top_k):
		# 	close_word = self.index2word[nearest[k].item()]
		# 	top_list.append(close_word)
		# return top_list


review_path = '/Users/yangsong/Desktop/Projects/gitrepo_songyang0716/NLP/word2vec_models/yelp_review_10000.txt'
w2v = Word2Vec(review_path)
w2v.train(epochs=1)
# print(w2v.corpus[0])







