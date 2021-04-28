import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import torchtext.legacy as legacy
from model import encoder, inner_attention


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
            val_hyp, val_hyp_length = val_batch.hypothesis
            val_prem, val_prem_length = val_batch.premise
            val_target = val_batch.label - 1
            scores = model(val_prem, val_hyp, val_prem_length, val_hyp_length)
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
num_layer = 1
p = 0.25
learning_rate = 0.001
batch_size = 128
hidden_sizes = 100
num_epochs = 5


for hidden_size in [hidden_sizes]:
    step = 0
    # define model structure
    sentence_embedding = encoder(hidden_size,
                                 pretrained_embeddings,
                                 embedding_size,
                                 num_layer,
                                 p).to(device)

    model = inner_attention(sentence_embedding).to(device)
    model.train()
    # define train iterator
    train_iterator, validation_iterator, test_iterator = \
        legacy.data.BucketIterator.splits((train_data,
                                          validation_data,
                                          test_data),
                                          batch_size=batch_size,
                                          device=device,
                                          sort_within_batch=True)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    writer = SummaryWriter('runs/SNLI/\
        inner_attention_tensorboard/hidden_size_{}'.format(hidden_size))
    if load_model:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"))

    # Train Network
    for epoch in range(num_epochs):
        losses = []
        accuracies = []
        if (epoch + 1) % 5 == 0:
            checkpoint = {'state_dict': model.state_dict(),
                          'optimizer': optimizer.state_dict()}
            save_checkpoint(checkpoint)
        for batch_idx, batch in enumerate(train_iterator):
            model.train()
            # data
            prem_sentences, prem_length = batch.premise
            hyp_sentences, hyp_length = batch.hypothesis

            # outcome data need to be between 0 - (n_class-1)
            target = batch.label - 1
            # forward
            scores = model(prem_sentences, hyp_sentences,
                           prem_length, hyp_length)
            loss = criterion(scores, target)
            losses.append(loss.item())

            # backward
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            _, predictions = scores.max(1)
            num_correct = (predictions == target).sum()
            running_train_acc = float(num_correct) / \
                float(hyp_sentences.shape[0])
            accuracies.append(running_train_acc)

            writer.add_scalar('Training Loss', loss, global_step=step)
            writer.add_scalar('Training Accuracy',
                              running_train_acc,
                              global_step=step)
            # step += 1
            model.eval()
            if batch_idx % 100 == 0:
                writer.add_scalar('Training Loss', loss, global_step=step)
                writer.add_scalar('Training Accuracy',
                                  running_train_acc,
                                  global_step=step)

                # Check for the running accuracy of validation
                check_accuracy(validation_iterator, model, step, writer)
                step += 1
