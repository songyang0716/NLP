import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F

class LSTM(nn.Module):
	def __init__(self, n_vocab, seq_size, emb_size, hidden_size):
		super(LSTM, self).__init__()
		self.seq_size = seq_size
		self.lstm_size = hidden_size
		self.embedding = nn.Embedding(n_vocab, emb_size)

		# if batch_first=True, input as (batch, seq, feature)
		self.lstm = nn.LSTM(input_size=emb_size,
							hidden_size=hidden_size,
							num_layers=1,
							batch_first=True)

		# from hidden layer to output layer
		# hidden_size to n_vocab
		self.dense = nn.Linear(hidden_size, n_vocab)

		# Xavier init
		self.embedding.weight.data = nn.init.xavier_uniform_(self.embedding.weight.data)
		# self.lstm.weight.data = nn.init.xavier_uniform_(self.lstm.weight.data)
		self.dense.weight.data = nn.init.xavier_uniform_(self.dense.weight.data)

		# self.embedding.weight.requires_grad = False

	def forward(self, x, prev_state):
		embed = self.embedding(x)
		# emd is of shape batch ,seq_len input_size
		# the prev_state contains two tensor, h_0 and c_0  
		# initial hidden state & initial cell state of shape (num_direction*layer , batch_size , hidden_size) 
		output, state = self.lstm(embed, prev_state)
		# raw output, batch_size * seq_length * n_vocab
		logits = self.dense(output)
		return logits, state
		# prob = self.softmax(logits)


	def initial_state(self, batch_size):
		h_0 = torch.zeros(1, batch_size, self.lstm_size)
		c_0 = torch.zeros(1, batch_size, self.lstm_size)
		return (h_0, c_0)







































