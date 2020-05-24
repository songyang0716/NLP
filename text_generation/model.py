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
		self.lstm.weight.data = nn.init.xavier_uniform_(self.lstm.weight.data)
		self.dense.weight.data = nn.init.xavier_uniform_(self.dense.weight.data)


	def forward(self, x, prev_state):
		embed = self.embedding(x)
		output, state = self.lstm(embed, prev_state)






































