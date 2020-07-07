# Reference code: https://github.com/coetaur0/ESIM

import torch
import torch.nn as nn
# import torchvision.transforms as T
import torch.nn.functional as F


def get_mask(sequences_batch, sequences_lengths):
	"""
	Get the mask for a batch of padded variable length sequences.
	Args:
		sequences_batch: A batch of padded variable length sequences
			containing word indices. Must be a 2-dimensional tensor of size
			(batch, sequence).
		sequences_lengths: A tensor containing the lengths of the sequences in
			'sequences_batch'. Must be of size (batch).
	Returns:
		A mask of size (batch, max_sequence_length), where max_sequence_length
		is the length of the longest sequence in the batch.
	"""
	batch_size = sequences_batch.size()[0]
	max_length = torch.max(sequences_lengths)
	mask = torch.ones(batch_size, max_length, dtype=torch.float)
	mask[sequences_batch[:, :max_length] == 0] = 0.0
	return mask

	
class Seq2SeqEncoder(nn.Module):
	"""
	RNN taking variable length padded sequences of vectors as input and
	encoding them into padded sequences of vectors of the same length.
	This module is useful to handle batches of padded sequences of vectors
	that have different lengths and that need to be passed through a RNN.
	The sequences are sorted in descending order of their lengths, packed,
	passed through the RNN, and the resulting sequences are then padded and
	permuted back to the original order of the input sequences.
	"""
	def __init__(self,
				 input_size,
				 hidden_size,
				 num_layers=1,
				 bias=True,
				 dropout=0.0,
				 bidirectional=False):
		"""
		Args:
			rnn_type: The type of RNN to use as encoder in the module.
				Must be a class inheriting from torch.nn.RNNBase
				(such as torch.nn.LSTM for example).
			input_size: The number of expected features in the input of the
				module.
			hidden_size: The number of features in the hidden state of the RNN
				used as encoder by the module.
			num_layers: The number of recurrent layers in the encoder of the
				module. Defaults to 1.
			bias: If False, the encoder does not use bias weights b_ih and
				b_hh. Defaults to True.
			dropout: If non-zero, introduces a dropout layer on the outputs
				of each layer of the encoder except the last one, with dropout
				probability equal to 'dropout'. Defaults to 0.0.
			bidirectional: If True, the encoder of the module is bidirectional.
				Defaults to False.
		"""
		super(Seq2SeqEncoder, self).__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.dropout = dropout
		self.bidirectional = bidirectional

		# if batch_first=True, input as (batch, seq, feature)
		self._lstm = nn.LSTM(input_size=self.input_dim,
							 hidden_size=self.hidden_dim,
							 num_layers=self.num_layers,
							 dropout=self.dropout,
							 batch_first=True,
							 bidirectional=self.bidirectional)


class ESIM(nn.Module):
	"""
		Implementation of CNN for classification
	"""
	def __init__(self, 
				 vocab_size,
				 embeddings_dim,
				 hidden_size,
				 embeddings,
				 dropout,
				 num_classes,
				 device):
		super(ESIM, self).__init__()

		self.vocab_size = vocab_size
		self.embeddings_dim = embeddings_dim
		self.hidden_size = hidden_size
		self.dropout = dropout
		self.num_classes = num_classes
		self.device = devie

		self._word_embedding = nn.Embedding(self.vocab_size,
											self.embedding_dim,
											padding_idx=0).requires_grad_(False)
		self._word_embedding.weight = nn.Parameter(embeddings)


		self._encoding = Seq2SeqEncoder(self.embedding_dim,
										self.hidden_size,
										bidirectional=True)


	def forward(self,
				premises,
				premises_lengths,
				hypotheses,
				hypotheses_lengths):
		"""
		Args:
			premises: A batch of varaible length sequences of word indices
				representing premises. The batch is assumed to be of size
				(batch, premises_length).
			premises_lengths: A 1D tensor containing the lengths of the
				premises in 'premises'.
			hypothesis: A batch of varaible length sequences of word indices
				representing hypotheses. The batch is assumed to be of size
				(batch, hypotheses_length).
			hypotheses_lengths: A 1D tensor containing the lengths of the
				hypotheses in 'hypotheses'.
		Returns:
			logits: A tensor of size (batch, num_classes) containing the
				logits for each output class of the model.
			probabilities: A tensor of size (batch, num_classes) containing
				the probabilities of each output class in the model.
		"""
		premises_mask = get_mask(premises, premises_lengths).to(self.device)
		hypotheses_mask = get_mask(hypotheses, hypotheses_lengths).to(self.device)

		embedded_premises = self._word_embedding(premises)
		embedded_hypotheses = self._word_embedding(hypotheses)
		
		encoded_premises = self._encoding(embedded_premises,
										  premises_lengths)
		encoded_hypotheses = self._encoding(embedded_hypotheses,
											hypotheses_lengths)
		

