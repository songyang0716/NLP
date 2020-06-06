import sys
import torch
import torch.optim as optim
import random
import numpy as np
import spacy
from collections import Counter
from nltk.tokenize import word_tokenize
# from model import LSTM 



# import argparse
# import os
# import time
# from progress.bar import Bar
# import yutils

import torch.nn as nn
from torch.autograd import Variable
# from nnet.blstm import BLSTM
# from nnet.lstm import LSTM
# from nnet.cnn import CNN



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



def main():
	embeddings_dict = {}

	in_dir = "./../../data/glove.6B/"
	with open(in_dir+"glove.6B.50d.txt", 'r', encoding="utf-8") as f:
		for line in f:
			values = line.split()
			word = values[0]
			vector = np.asarray(values[1:], "float32")
			embeddings_dict[word] = vector
	# print(len(embeddings_dict))
    emb_np = numpy.asarray(embeddings, dtype=numpy.float32)  # from_numpy
    emb = torch.from_numpy(emb_np)



if __name__ == '__main__':
	main()

