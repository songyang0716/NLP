import torch.nn as nn
import torch.optim as optim

from torchtext.data import Field
from torchtext.data import BucketIterator

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
    device='cpu')

# Set device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Hyperparameter
input_size = 50
num_classes = 3
num_layers = 1
hidden_size = 256
embedding_size = 50
learning_rate = 0.001
batch_size = 64
num_epochs = 2

# input_size, hidden_size, num_classes, embeddings
# Initialize network
model = Sentence_RNN(input_size,
    hidden_size,
    num_classes,
    inputs.vocab.vectors)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(train_iterator):
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
        if batch_idx // 100 == 0:
            print(loss)

        # gradient update
        optimizer.step()


# pick one batch, and try to overfit this batch
# (premise_first, hypothesis_first, label_first), _ = next(iter(train_iterator))
# # the label is from 1-3
# label_first = label_first-1


# train_data[0].__dict__.values()
# train_data[0].__dict__.keys()

# embedding_layer = nn.Embedding.from_pretrained(TEXT.vocab.vectors)

# # Or if you want to make it trainable
# trainable_embedding_layer = nn.Embedding.from_pretrained(TEXT.vocab.vectors, freeze=False)
