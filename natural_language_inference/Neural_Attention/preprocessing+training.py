import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import torchtext.legacy as legacy
from model import encoder1, encoder2, attention


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set up fields, hypothesis + premise and categories
inputs = legacy.data.Field(lower=True, tokenize='spacy', batch_first=True)
answers = legacy.data.Field(sequential=False)

# make splits for data
train_data, validation_data, test_data = \
    legacy.datasets.SNLI.splits(text_field=inputs, label_field=answers)

# build the vocabulary
inputs.build_vocab(train_data, min_freq=1, vectors='glove.6B.300d')
answers.build_vocab(train_data)

pretrained_embeddings = inputs.vocab.vectors


# Hyperparameter
num_classes = 3
num_layers = 1
# hidden_size = 159
embedding_size = 300
# learning_rate = 0.001
# batch_size = 64
p = 0.5


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
        print("Val set accuracy: {}".format(acc))
    return acc


learning_rates = [0.001, 0.01]
batch_sizes = [64, 256]
hidden_sizes = [100, 300, 500]
num_epochs = 20

for learning_rate in learning_rates:
    for batch_size in batch_sizes:
        for hidden_size in hidden_sizes:
            step = 0
            # define model structure
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
            model.train()
            # define train iterator
            train_iterator, validation_iterator, test_iterator = \
                legacy.data.BucketIterator.splits((train_data,
                                                  validation_data,
                                                  test_data),
                                                  batch_size=batch_size,
                                                  device=device)
            # Loss and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            writer = SummaryWriter('runs/SNLI/attention_tensorboard/LR_{} batch_size_{} hidden_size_{}'.format(learning_rate, batch_size, hidden_size))
            # Train Network
            for epoch in range(num_epochs):
                losses = []
                accuracies = []

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
                    losses.append(loss.item())

                    # backward
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm(model.parameters(), max_norm=1)
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
                    step += 1

        # model.eval()
        # if batch_idx % 1000 == 0:
        #     # Check for the running accuracy of training
        #     print(loss)
        #     _, predictions = scores.max(1)
        #     num_correct = (predictions == target).sum()
        #     print("Training set accuracy: {}".format(num_correct/batch_size))

        #     # Check for the running accuracy of validation
        #     check_accuracy(validation_iterator, model)
# %tensorboard --logdir ./runs
