import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F

class selfAttentive(nn.Module):
	"""
		Implementation of SELF-ATTENTIVE SENTENCE EMBEDDING for sentiment classification task
	"""
	def __init__(self, embeddings, input_dim, hidden_dim, num_layers, output_dim, da_dim, max_len, dropout):
		super(selfAttentive, self).__init__()
		
		self.max_len = max_len
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim
		self.da_dim = da_dim

		# Initial the embedding with glove
		# https://stackoverflow.com/questions/61172400/what-does-padding-idx-do-in-nn-embeddings
		self.emb = nn.Embedding(embeddings.size(0), embeddings.size(1), padding_idx=0).requires_grad_(False)
		self.emb.weight = nn.Parameter(embeddings)

		# if batch_first=True, input as (batch, seq, feature)
		self.lstm = nn.LSTM(input_size=input_dim,
							hidden_size=hidden_dim,
							num_layers=num_layers,
							dropout=dropout,
							batch_first=True,
				 			bidirectional=True)

		self.W_s1 = nn.Linear(2*self.hidden_dim, self.da_dim)
		self.W_s2 = nn.Linear(self.da_dim, 30)

		# because of bidirectional LSTM, so our output layer has dimension of 2*hidden
		self.output = nn.Linear(2*self.hidden_dim, output_dim)

	def attention_net(self, lstm_output):
		# use da = 350, r = 30 & penalization_coeff = 1 as per given in the self-attention original ICLR paper
		attn_weight_matrix = self.W_s2(F.tanh(self.W_s1(lstm_output)))
		A = F.softmax(attn_weight_matrix, dim=1)
		return A 

	def forward(self, sen_batch, sen_lengths):
		"""
		:param sen_batch: shape: (batch, sen_length), tensor for sentence sequence
		:param sen_lengths:
		:return:
		"""

		''' Embedding Layer | Padding | Sequence_length 40'''
		sen_batch = self.emb(sen_batch)
		batch_size = len(sen_batch)
		''' Bi-LSTM Computation '''
		# If (h_0, c_0) is not provided, both h_0 and c_0 default to zero.
		sen_outs, _ = self.lstm(sen_batch.view(batch_size, -1, self.input_dim))
		# sen_outs is with shape: batch size * seq length * 2hidden_dim

		attn_weight_matrix = self.attention_net(sen_outs)
		# attn is with shape : batch size * seq length * r
		attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)

		hidden_matrix = torch.bmm(attn_weight_matrix, sen_outs)
		# print(hidden_matrix.shape)








