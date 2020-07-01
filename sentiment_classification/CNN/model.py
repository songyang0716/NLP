import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F

class CNN(nn.Module):
	"""
		Implementation of CNN for classification
	"""
	def __init__(self, embeddings, input_dim, window_dim, filter_dim, output_dim, max_len, dropout):
		super(CNN, self).__init__()
		
		"""
		Arguments
		---------
		embeddings : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table
		--------
		
		"""

		self.max_len = max_len
		self.input_dim = input_dim
		self.window_dim = window_dim
		self.filter_dim = filter_dim

		# Initial the embedding with glove
		# padding_idx means if the input is with index 0, then padding with zero vectors
		# https://stackoverflow.com/questions/61172400/what-does-padding-idx-do-in-nn-embeddings
		# print(embeddings.size(0))
		# print(embeddings.size(1))
		self.emb = nn.Embedding(embeddings.size(0), embeddings.size(1), padding_idx=0).requires_grad_(False)
		self.emb.weight = nn.Parameter(embeddings)
		# print(self.emb.size())

		self.cnn = nn.Conv2d(in_channels=1,
							 out_channels=filter_dim,
							 kernel_size=(window_dim, embeddings.size(1)),
							 stride=1,
							 padding=0)
		
