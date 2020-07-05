# Reference code: https://github.com/coetaur0/ESIM

import torch
import torch.nn as nn
# import torchvision.transforms as T
import torch.nn.functional as F

class ESIM(nn.Module):
	"""
		Implementation of CNN for classification
	"""
	def __init__(self, 
				 embeddings_size,
				 embeddings_dim,
				 hidden_size,
				 embeddings,
				 dropout,
				 num_classes,
				 device):
		super(ESIM, self).__init__()