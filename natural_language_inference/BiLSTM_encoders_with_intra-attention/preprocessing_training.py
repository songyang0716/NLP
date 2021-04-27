import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import torchtext.legacy as legacy
from model import encoder


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set up fields, hypothesis + premise and categories
# spacy tokenizer is slow !!!, could use default one split
inputs = legacy.data.Field(init_token='</s>',
                           lower=True,
                           batch_first=True,
                           include_lengths=True)
answers = legacy.data.Field(sequential=False)

# make splits for data
train_data, validation_data, test_data = \
    legacy.datasets.SNLI.splits(text_field=inputs, label_field=answers)


# build the vocabulary
inputs.build_vocab(train_data, min_freq=1, vectors='glove.6B.300d')
answers.build_vocab(train_data)

pretrained_embeddings = inputs.vocab.vectors


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


# Check for the accuracy on the test and val to see the model performance
def check_accuracy(validation_iterator, model, step, writer):
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
        writer.add_scalar('Validation set accuracy', acc, global_step=step)
        print("Val set accuracy: {}".format(acc))
    return acc


# Parameters
num_classes = 3
embedding_size = 300
load_model = False

# Model hyperparameter
num_layers = 1
p = 0.25
learning_rates = 0.001
batch_sizes = 128
hidden_sizes = 100
num_epochs = 5


