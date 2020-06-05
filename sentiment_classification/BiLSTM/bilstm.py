import sys
import torch
import torch.optim as optim
import random
import numpy as np
import spacy
from collections import Counter
from nltk.tokenize import word_tokenize
# from model import LSTM 




def train(model, training_data, args, optimizer, criterion):
    model.train()

    batch_size = args.batch_size

    sentences, sentences_seqlen, sentences_mask, labels = training_data

    # print batch_size, len(sentences), len(labels)

    assert batch_size == len(sentences) == len(labels)

    ''' Prepare data and prediction'''
    sentences_, sentences_seqlen_, sentences_mask_ = \
        var_batch(args, batch_size, sentences, sentences_seqlen, sentences_mask)
    labels_ = Variable(torch.LongTensor(labels))
    if args.cuda:
        labels_ = labels_.cuda()

    assert len(sentences) == len(labels)

    model.zero_grad()
    probs = model(sentences_, sentences_seqlen_, sentences_mask_)
    loss = criterion(probs.view(len(labels_), -1), labels_)

    loss.backward()
    optimizer.step()

    