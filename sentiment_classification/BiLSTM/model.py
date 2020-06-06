import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F

class BLSTM(nn.Module):
	def __init__(self, n_vocab, seq_size, emb_size, hidden_size):
		super(BLSTM, self).__init__()


	def forward(self, x, prev_state):



	def initial_state(self, batch_size):
		h_0 = torch.zeros(1, batch_size, self.lstm_size)
		c_0 = torch.zeros(1, batch_size, self.lstm_size)
		return (h_0, c_0)



