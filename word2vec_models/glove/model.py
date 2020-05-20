import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
import torchvision.transforms as T
import torch.nn.functional as F


class GloVe(nn.Module):
        """
        :param co_oc: Co-occurrence ndarray with shape of [num_classes, num_classes]
        :param embed_size: embedding size
        :param x_max: An int representing cutoff of the weighting function
        :param alpha: Ant float parameter of the weighting function
        """

	def __init__(self, co_oc, emb_dim, x_max=100, alpha=0.75):
		super(GloVe, self).__init__()

		self.emb_dim = emb_dim
		self.x_max = x_max
		self.alpha = alpha

		# to prevent the log(0) issue
		self.co_oc = self.co_oc + 1
		self.num_classes = self.co_oc.shape[0]

		self.input_emb = nn.Embedding(self.num_classes, self.emb_dim)
		self.output_emb = nn.Embedding(self.num_classes, self.emb_dim)

		self.input_bias = nn.Embedding(self.num_classes,1)
		self.output_bias = nn.Embedding(self.num_classes,1)

		# Xavier init
		self.input_emb.weight.data = nn.init.xavier_uniform_(self.input_emb.weight.data)
		self.output_emb.weight.data = nn.init.xavier_uniform_(self.output_emb.weight.data)

		self.input_bias.weight.data = nn.init.xavier_uniform_(self.input_bias.weight.data)
		self.output_bias.weight.data = nn.init.xavier_uniform_(self.output_bias.weight.data)


	def forward(self, input_idx, output_idx):
        """
        :param input: An array with shape of [batch_size] of int type
        :param output: An array with shape of [batch_size] of int type
        :return: loss estimation for Global Vectors word representations
                 defined in nlp.stanford.edu/pubs/glove.pdf
        """
        batch_size = len(input_idx)
        input_batch_embed = self.input_emb[input_idx]
        input_batch_bias = self.input_bias[input_idx]

        output_batch_embed = self.output_emb[output_idx]
        output_batch_bias = self.output_bias[output_idx]

        output_matrix = torch.sum(torch.mul(input_batch_embed, output_batch_embed), dim=1) + input_batch_bias + output_batch_bias - self.co_oc[zip(input_idx, output_idx)]
        


