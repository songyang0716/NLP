import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import torchtext.legacy as legacy
from model import encoder1, encoder2, attention


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set up fields, hypothesis + premise and categories
# spacy tokenizer is slow !!!, could use default one split
inputs = legacy.data.Field(init_token='</s>',
                           lower=True,
                           tokenize='spacy',
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


# Hyperparameter
num_classes = 3
num_layers = 1
embedding_size = 300
p = 0.5
load_model = False


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


learning_rates = [0.001]
batch_sizes = [128]
hidden_sizes = [100]
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

            model = attention(premise_encoder, hyp_encoder, hidden_size).to(device)
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
            writer = SummaryWriter('runs/SNLI/attention_tensorboard/LR_{} batch_size_{} hidden_size_{}'.format(learning_rate, batch_size, hidden_size))
            if load_model:
                load_checkpoint(torch.load("my_checkpoint.pth.tar"))

            # Train Network
            for epoch in range(num_epochs):
                losses = []
                accuracies = []
                if (epoch+1) % 10 == 0:
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
                        writer.add_scalar('Training Accuracy', running_train_acc, global_step=step)

                        # Check for the running accuracy of validation
                        check_accuracy(validation_iterator, model, step, writer)
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

# Load model checkpoint

