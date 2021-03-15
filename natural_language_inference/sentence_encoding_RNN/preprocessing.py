import time
import pickle

import numpy as np
import random
from collections import Counter

import spacy
import torch

from torchtext.data import Field
from torchtext.data import BucketIterator
from torchtext.data import TabularDataset

from torchtext.datasets import SNLI
from model import Sentence_RNN

# set up fields, hypothesis + premise and categories
inputs = Field(lower=True, tokenize='spacy', batch_first=True)
answers = Field(sequential=False)

# make splits for data
train_data, validation_data, test_data = SNLI.split(fields=(inputs, answers))

# build the vocabulary
inputs.build_vocab(train_data, min_freq=2, vectors='glove.6B.50d')
answers.build_vocab(train_data)

# Embedding dataset 
#inputs.vocab.vectors.size()
# Index of the word "the"

# word => index
#TEXT.vocab.stoi["the"] # => 2

pretrained_embeddings = inputs.vocab.vectors

train_iterator, validation_iterator, test_iterator = BucketIterator.splits(
    (train_data, validation_data, test_data),
    batch_size=64,
    device='cuda')


# train_data[0].__dict__.values()
# train_data[0].__dict__.keys()

# embedding_layer = nn.Embedding.from_pretrained(TEXT.vocab.vectors)

# # Or if you want to make it trainable
# trainable_embedding_layer = nn.Embedding.from_pretrained(TEXT.vocab.vectors, freeze=False)


model = 














