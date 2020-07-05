# Reference code: https://github.com/coetaur0/ESIM
import numpy as np
import random
import pickle
import os 

import torch 
import torch.nn as nn 
from preprocessing import NLIDataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def main(train_file,
		 valid_file,
		 embeddings_file,
		 target_dir,
		 hidden_size=300,
		 dropout=0.5,
		 num_classes=3,
		 epochs=64,
		 batch_size=32,
		 lr=0.0004,
		 patience=5,
		 max_grad_norm=10.0,
		 checkpoint=None):
	"""
	Train the ESIM model on the SNLI dataset.
	Args:
		train_file: A path to some preprocessed data that must be used
			to train the model.
		valid_file: A path to some preprocessed data that must be used
			to validate the model.
		embeddings_file: A path to some preprocessed word embeddings that
			must be used to initialise the model.
		target_dir: The path to a directory where the trained model must
			be saved.
		hidden_size: The size of the hidden layers in the model. Defaults
			to 300.
		dropout: The dropout rate to use in the model. Defaults to 0.5.
		num_classes: The number of classes in the output of the model.
			Defaults to 3.
		epochs: The maximum number of epochs for training. Defaults to 64.
		batch_size: The size of the batches for training. Defaults to 32.
		lr: The learning rate for the optimizer. Defaults to 0.0004.
		patience: The patience to use for early stopping. Defaults to 5.
		checkpoint: A checkpoint from which to continue training. If None,
			training starts from scratch. Defaults to None.
	"""
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	print(20 * "=", " Preparing for training ", 20 * "=")
	# -------------------- Data loading ------------------- #
	print("\t* Loading training data...")
	with open(train_file, "rb") as pkl:
		train_data = NLIDataset(pickle.load(pkl))
	print("Training data length: ", len(train_data))

	train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

	print("\t* Loading validation data...")
	with open(valid_file, "rb") as pkl:
		valid_data = NLIDataset(pickle.load(pkl))
	print("Validation data length: ", len(valid_data))


	valid_loader = DataLoader(valid_data, shuffle=False, batch_size=batch_size)

	print(train_loader)
	


if __name__ == '__main__':
	main(train_file='./../data/train_data.pkl',
		 valid_file='./../data/dev_data.pkl',
		 embeddings_file='./../data/embeddings.pkl',
		 target_dir='./../data/')













