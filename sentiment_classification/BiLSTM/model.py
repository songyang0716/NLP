import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F

class BLSTM(nn.Module):
	"""
		Implementation of BLSTM Concatenation for sentiment classification task
	"""
	def __init__(self, embeddings, input_dim, hidden_dim, num_layers, output_dim, max_len, dropout):
		super(BLSTM, self).__init__()
		
		self.sen_length = max_len
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim

		# Initial the embedding with glove
		# padding_idx means if the input is with index 0, then padding with zero vectors
		# https://stackoverflow.com/questions/61172400/what-does-padding-idx-do-in-nn-embeddings
		self.emb = nn.Embedding(embeddings.size(0), embeddings.size(1), padding_idx=0)
		self.emb.weight = nn.Parameter(embeddings)

		# if batch_first=True, input as (batch, seq, feature)
		self.lstm = nn.LSTM(input_size=input_dim,
							hidden_size=hidden_dim,
							num_layers=num_layers,
							dropout=dropout,
							batch_first=True,
							bidirectional=True)

		# because of bidirectional LSTM, so our output layer has dimension of 2*hidden
		self.output = nn.Linear(2*self.hidden_dim, output_dim)

	def bi_fetch(self, rnn_outs, seq_lengths, batch_size, max_len):
		"""
		param run_outs: output from the LSTM bidirectional function
		param seq_lengths
		param batch_size
		param max_len
		"""
		rnn_outs = rnn_outs.view(batch_size, max_len, 2, -1)


	def forward(self, sen_batch, sen_lengths):
		"""
		:param sen_batch: (batch, sen_length), tensor for sentence sequence
		:param sen_lengths:
		:return:
		"""

		''' Embedding Layer | Padding | Sequence_length 40'''
		sen_batch = self.emb(sen_batch)
		batch_size = len(sen_batch)

		''' Bi-LSTM Computation '''
		# If (h_0, c_0) is not provided, both h_0 and c_0 default to zero.
		sen_outs, _ = self.lstm(sen_batch.view(batch_size, -1, self.input_dim))
		sen_rnn = sen_outs.view(batch_size, -1, 2 * self.hidden_dim)
		''' 
		Fetch the truly last hidden layer of both sides
		'''
		print(sen_rnn.size())

		# difference between sen_lengths and self.sen_length ? 
		# sentence_batch = self.bi_fetch(sen_rnn, sen_lengths, batch_size, self.sen_length) 
		# out = self.output(sentence_batch)
		# out_prob = F.softmax(out.view(batch_size, -1))
		# return out_prob



	# def initial_state(self, batch_size):
		# h_0 = torch.zeros(1, batch_size, self.hidden_dim)
		# c_0 = torch.zeros(1, batch_size, self.hidden_dim)
		# return (h_0, c_0)



