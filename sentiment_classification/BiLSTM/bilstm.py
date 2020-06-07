import sys
import torch
import torch.nn as nn
# from torch.autograd import Variable
import torch.optim as optim
import random
import numpy as np
from model import BLSTM 




embsize = 50
hidden_size = 64
n_layers = 1
max_len = 20
dropout = 0
l_rate = 0.01
input_dir = "./data/"


# def var_batch(args, batch_size, sentences, sentences_seqlen, sentences_mask):
#     """
#     Transform the input batch to PyTorch variables
#     :return:
#     """
#     # dtype = torch.from_numpy(sentences, dtype=torch.cuda.LongTensor)
#     sentences_ = Variable(torch.LongTensor(sentences).view(batch_size, args.sen_max_len))
#     sentences_seqlen_ = Variable(torch.LongTensor(sentences_seqlen).view(batch_size, 1))
#     sentences_mask_ = Variable(torch.LongTensor(sentences_mask).view(batch_size, args.sen_max_len))

#     if args.cuda:
#         sentences_ = sentences_.cuda()
#         sentences_seqlen_ = sentences_seqlen_.cuda()
#         sentences_mask_ = sentences_mask_.cuda()

#     return sentences_, sentences_seqlen_, sentences_mask_





# def train(model, training_data, args, optimizer, criterion):
#     model.train()

#     batch_size = args.batch_size

#     sentences, sentences_seqlen, sentences_mask, labels = training_data

#     # print batch_size, len(sentences), len(labels)

#     assert batch_size == len(sentences) == len(labels)

#     ''' Prepare data and prediction'''
#     sentences_, sentences_seqlen_, sentences_mask_ = \
#         var_batch(args, batch_size, sentences, sentences_seqlen, sentences_mask)
#     labels_ = Variable(torch.LongTensor(labels))
#     if args.cuda:
#         labels_ = labels_.cuda()

#     assert len(sentences) == len(labels)

#     model.zero_grad()
#     probs = model(sentences_, sentences_seqlen_, sentences_mask_)
#     loss = criterion(probs.view(len(labels_), -1), labels_)

#     loss.backward()
#     optimizer.step()



##############################
#   Serialization to pickle  #
##############################
def pickle2dict(in_file):
	try:
		import cPickle as pickle
	except ImportError:
		import pickle
	with open(in_file, 'rb') as f:
		your_dict = pickle.load(f)
		return your_dict


def main():
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(device)

	dataset = pickle2dict(input_dir + "features_glove.pkl")
	embeddings = pickle2dict(input_dir + "embeddings_glove.pkl")
	dataset["embeddings"] = embeddings

	emb_np = np.asarray(embeddings, dtype=np.float32)
	emb = torch.from_numpy(emb_np)


	blstm_models = BLSTM(embeddings=emb,
						input_dim=embsize,
						hidden_dim=hidden_size,
						num_layers=n_layers,
						output_dim=2,
						max_len=max_len,
						dropout=dropout)

	blstm_models = blstm_models.to(device)

	optimizer = optim.SGD(blstm_models.parameters(), lr=l_rate, weight_decay=1e-5)
	criterion = nn.CrossEntropyLoss()

	training_set = dataset["training"]
	# print(blstm_models)


if __name__ == '__main__':
	main()






























