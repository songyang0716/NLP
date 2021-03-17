import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.data import Field
from torchtext.data import BucketIterator

from torchtext.datasets import SNLI
from model import Sentence_RNN, Sentence_LSTM

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

# Embedding dataset
# inputs.vocab.vectors.size()
# Index of the word "the"

# word => index
# TEXT.vocab.stoi["the"] # => 2

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
num_epochs = 100

# input_size, hidden_size, num_classes, embeddings
# Initialize network
# model = Sentence_RNN(hidden_size,
#                      num_classes,
#                      inputs.vocab.vectors,
#                      embedding_size,
#                      num_layers).to(device)
model = Sentence_LSTM(hidden_size,
                      num_classes,
                      inputs.vocab.vectors,
                      embedding_size,
                      num_layers).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Check for the accuracy on the test and val to see the model performance
def check_accuracy(validation_iterator, model):
    num_correct = 0
    num_sample = 0

    with torch.no_grad():
        for val_batch_idx, val_batch in enumerate(validation_iterator):
            val_hyp = val_batch.hypothesis
            val_prem = val_batch.premise

            val_target = val_batch.label - 1
            scores = model(val_hyp, val_prem)
            # return the indices of each prediction
            _, predictions = scores.max(1)
            num_correct += (predictions == val_target).sum()
            num_sample += predictions.size(0)
        acc = (num_correct / num_sample)
        print("The val set accuracy is {}".format(acc))
    return acc


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

        model.eval()
        if batch_idx % 1000 == 0:
            print(loss)
            check_accuracy(validation_iterator, model)


# pick one batch, and try to overfit this batch
# (premise_first, hypothesis_first, label_first), _ = next(iter(train_iterator))
# # the label is from 1-3
# label_first = label_first-1


# train_data[0].__dict__.values()
# train_data[0].__dict__.keys()

# embedding_layer = nn.Embedding.from_pretrained(TEXT.vocab.vectors)

# # Or if you want to make it trainable
# trainable_embedding_layer = nn.Embedding.from_pretrained(TEXT.vocab.vectors, freeze=False)
