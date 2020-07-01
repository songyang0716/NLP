import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F

class CNN(nn.Module):
	"""
		Implementation of CNN for classification
	"""