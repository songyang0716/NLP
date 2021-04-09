import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.data import Field
from torchtext.data import BucketIterator

from torchtext.datasets import SNLI
from model import encoder1, encoder2, attention


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set up fields, hypothesis + premise and categories
inputs = Field(lower=True, tokenize='spacy', batch_first=True)
answers = Field(sequential=False)

# make splits for data
train_data, validation_data, test_data = SNLI.splits(text_field=inputs, label_field=answers)

# build the vocabulary
inputs.build_vocab(train_data, min_freq=2, vectors='glove.6B.300d')
answers.build_vocab(train_data)

pretrained_embeddings = inputs.vocab.vectors

train_iterator, validation_iterator, test_iterator = BucketIterator.splits(
    (train_data, validation_data, test_data),
    batch_size=64,
    device=device)


# Hyperparameter
num_classes = 3
num_layers = 1
hidden_size = 100
embedding_size = 300
learning_rate = 0.001
batch_size = 64
num_epochs = 1
p = 0.5

premise_encoder = encoder1(hidden_size,
                           pretrained_embeddings,
                           embedding_size,
                           num_layers,
                           p).to(device)

hyp_encoder = encoder2(hidden_size,
                       pretrained_embeddings,
                       embedding_size,
                       num_layers,
                       p).to(device)

model = attention(premise_encoder, hyp_encoder).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Train Network
for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(train_iterator):
        model.train()
        # data
        hyp_sentences = batch.hypothesis
        prem_sentences = batch.premise

        # outcome data need to be between 0 - (n_class-1)
        target = batch.label - 1
        # forward
        scores = model(hyp_sentences, prem_sentences)
        loss = criterion(scores, target)

        # backward
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), max_norm=1)

        # gradient update
        optimizer.step()

        break
