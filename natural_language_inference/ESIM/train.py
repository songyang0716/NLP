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

from model import ESIM
from tqdm import tqdm


def train(model,
		  dataloader,
		  optimizer,
		  criterion,
		  epoch_number,
		  max_gradient_norm):
	"""
	Train a model for one epoch on some input data with a given optimizer and
	criterion.
	Args:
		model: A torch module that must be trained on some input data.
		dataloader: A DataLoader object to iterate over the training data.
		optimizer: A torch optimizer to use for training on the input model.
		criterion: A loss criterion to use for training.
		epoch_number: The number of the epoch for which training is performed.
		max_gradient_norm: Max. norm for gradient norm clipping.
	Returns:
		epoch_time: The total time necessary to train the epoch.
		epoch_loss: The training loss computed for the epoch.
		epoch_accuracy: The accuracy computed for the epoch.
	"""
	# Switch the model to train mode.
	model.train()
	device = model.device

	epoch_start = time.time()
	batch_time_avg = 0.0
	running_loss = 0.0
	correct_preds = 0
	# tqdm_batch_iterator = tqdm(dataloader)

	for batch_index, batch in enumerate(dataloader):
		batch_start = time.time()

		# Move input and output data to the GPU if it is used.
		premises = batch["premise"].to(device)
		premises_lengths = batch["premise_length"].to(device)
		hypotheses = batch["hypothesis"].to(device)
		hypotheses_lengths = batch["hypothesis_length"].to(device)
		labels = batch["label"].to(device)

		optimizer.zero_grad()

		logits, probs = model(premises,
							  premises_lengths,
							  hypotheses,
							  hypotheses_lengths)
		
		loss = criterion(logits, labels)
		loss.backward()

		# nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
		optimizer.step()
		batch_time_avg += time.time() - batch_start
		running_loss += loss.item()
		correct_preds += correct_predictions(probs, labels)


	epoch_time = time.time() - epoch_start
	epoch_loss = running_loss / len(dataloader)
	epoch_accuracy = correct_preds / len(dataloader.dataset)

	return epoch_time, epoch_loss, epoch_accuracy

	
def main(train_file,
		 valid_file,
		 embeddings_file,
		 target_dir,
		 hidden_size=300,
		 dropout=0.5,
		 num_classes=3,
		 epochs=50,
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

	# print(train_loader)
	# -------------------- Model definition ------------------- #
	print("\t* Building model...")
	with open(embeddings_file, "rb") as pkl:
		embeddings = torch.tensor(pickle.load(pkl), dtype=torch.float)\
					 .to(device)
					 
	print(embeddings.size())
	
	esim_model = ESIM(embeddings.shape[0],
					  embeddings.shape[1],
					  hidden_size,
					  embeddings,
					  dropout,
					  num_classes,
					  device).to(device)

	# -------------------- Preparation for training  ------------------- #
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(esim_model.parameters(), lr=l_rate, weight_decay=1e-5)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
														   mode="max",
														   factor=0.5,
														   patience=0)

	best_score = 0.0
	start_epoch = 1

	# Data for loss curves plot.
	epochs_count = []
	train_losses = []
	valid_losses = []

	# -------------------- Training epochs ------------------- #
	print("\n",
		  20 * "=",
		  "Training ESIM model on device: {}".format(device),
		  20 * "=")


	patience_counter = 0
	for epoch in range(start_epoch, epochs+1):
		epochs_count.append(epoch)

		print("* Training epoch {}:".format(epoch))
		epoch_time, epoch_loss, epoch_accuracy = train(esim_model,
													   train_loader,
													   optimizer,
													   criterion,
													   epoch,
													   max_grad_norm)



		train_losses.append(epoch_loss)
		print("-> Training time: {:.4f}s, loss = {:.4f}, accuracy: {:.4f}%"
			  .format(epoch_time, epoch_loss, (epoch_accuracy*100)))
		# description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}"\
		#               .format(batch_time_avg/(batch_index+1),
		#                       running_loss/(batch_index+1))
		# tqdm_batch_iterator.set_description(description)	


if __name__ == '__main__':
	main(train_file='./../data/train_data.pkl',
		 valid_file='./../data/dev_data.pkl',
		 embeddings_file='./../data/embeddings.pkl',
		 target_dir='./../data/')













