import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import torchtext.legacy as legacy
# from model import encoder, inner_attention
import model

# import EarlyStopping
from pytorchtools import EarlyStopping


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set up fields, hypothesis + premise and categories
# spacy tokenizer is slow !!!, could use default one split
inputs = legacy.data.Field(
    init_token="</s>", lower=True, batch_first=True, include_lengths=True
)
answers = legacy.data.Field(sequential=False)

# make splits for data
train_data, validation_data, test_data = legacy.datasets.SNLI.splits(
    text_field=inputs, label_field=answers
)

# build the vocabulary
inputs.build_vocab(train_data, min_freq=1, vectors="glove.6B.300d")
answers.build_vocab(train_data)

pretrained_embeddings = inputs.vocab.vectors

train_iterator, validation_iterator, test_iterator = \
    legacy.data.BucketIterator.splits((train_data,
                                      validation_data,
                                      test_data),
                                      batch_size=batch_size,
                                      device=device,
                                      sort_within_batch=True)


model = DirectionalSelfAttention(pretrained_embeddings, 300, 100)

for batch_idx, batch in enumerate(train_iterator):
    prem_sentences, prem_length = batch.premise
    hyp_sentences, hyp_length = batch.hypothesis
    # outcome data need to be between 0 - (n_class-1)
    target = batch.label - 1

    scores = model(prem_sentences)
    break