import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import torchtext.legacy as legacy
from model import encoder, inner_attention

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


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


def check_accuracy(validation_iterator, model, criterion):
    """
      Check for the accuracy on the val to see the model performance
    """
    val_losses = []
    val_accuracies = []
    with torch.no_grad():
        for val_batch_idx, val_batch in enumerate(validation_iterator):
            val_hyp, val_hyp_length = val_batch.hypothesis
            val_prem, val_prem_length = val_batch.premise
            val_target = val_batch.label - 1
            scores = model(val_prem, val_hyp, val_prem_length, val_hyp_length)
            loss = criterion(scores, val_target)
            # return the indices of each prediction
            _, predictions = scores.max(1)
            num_correct = float((predictions == val_target).sum())
            num_sample = float(predictions.size(0))
            val_losses.append(loss.item())
            val_accuracies.append(num_correct / num_sample)
    return val_losses, val_accuracies


# Parameters
num_classes = 3
embedding_size = 300
load_model = False

# Model hyperparameter
num_layer = 1
p = 0.25
learning_rates = [0.0001, 0.00005]
batch_size = 128
hidden_sizes = [300]
num_epochs = 10


for hidden_size in hidden_sizes:
    for learning_rate in learning_rates:
        step = 0
        # define model structure
        sentence_embedding = encoder(
            hidden_size, pretrained_embeddings, embedding_size, num_layer, p
        ).to(device)

        model = inner_attention(sentence_embedding).to(device)
        # model.train()
        # define train iterator
        (
            train_iterator,
            validation_iterator,
            test_iterator,
        ) = legacy.data.BucketIterator.splits(
            (train_data, validation_data, test_data),
            batch_size=batch_size,
            device=device,
            sort_within_batch=True,
        )
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        writer = SummaryWriter(
            "runs/SNLI/\
            inner_attention_tensorboard/hidden_size_{} learning_rate_{}".format(
                hidden_size, learning_rate
            )
        )
        if load_model:
            load_checkpoint(torch.load("my_checkpoint.pth.tar"))

        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=5, verbose=True)

        # to track the average training loss per epoch as the model trains
        avg_train_losses = []
        # to track the average validation loss per epoch as the model trains
        avg_valid_losses = []

        # to track the average training accuracy per epoch as the model trains
        avg_train_accuracy = []
        # to track the average validation accuracy per epoch as the model trains
        avg_valid_accuracy = []

        # Train Network
        for epoch in range(num_epochs):
            ###############
            # Train model #
            ###############
            model.train()
            train_losses = []
            train_accuracies = []

            if (epoch + 1) % 5 == 0:
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                save_checkpoint(checkpoint)

            for batch_idx, batch in enumerate(train_iterator):
                # clear the gradients of all optimized variables
                optimizer.zero_grad()
                # data
                prem_sentences, prem_length = batch.premise
                hyp_sentences, hyp_length = batch.hypothesis

                # outcome data need to be between 0 - (n_class-1)
                target = batch.label - 1
                # forward
                scores = model(
                    prem_sentences, hyp_sentences, prem_length, hyp_length
                )
                loss = criterion(scores, target)
                train_losses.append(loss.item())

                # backward
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()

                _, predictions = scores.max(1)
                num_correct = (predictions == target).sum()
                running_train_acc = float(num_correct) / float(
                    hyp_sentences.shape[0]
                )
                train_accuracies.append(running_train_acc)

            writer.add_scalar(
                "Training Loss", np.mean(train_losses), global_step=step
            )

            writer.add_scalar(
                "Training Accuracy", np.mean(train_accuracies), global_step=step
            )
            # step += 1
            model.eval()
            # Check for the running accuracy of validation
            val_losses, val_accuracies = \
                check_accuracy(validation_iterator, model, criterion)

            writer.add_scalar(
                "Validation Loss", np.mean(val_losses), global_step=step
            )
            writer.add_scalar(
                "Validation Accuracy",
                np.mean(val_accuracies),
                global_step=step,
            )

            avg_train_losses.append(np.mean(train_losses))
            avg_valid_losses.append(np.mean(val_losses))

            avg_train_accuracy.append(np.mean(train_accuracies))
            avg_valid_accuracy.append(np.mean(val_accuracies))

            step += 1

            print_msg = (f'[{epoch+1}/{num_epochs+1}] ' +
                         f'train_accur: {np.mean(train_accuracies):.3f} ' +
                         f'valid_accur: {np.mean(val_accuracies):.3f}')

            print(print_msg)

            early_stopping(np.mean(val_losses), model)

            if early_stopping.early_stop:
                print("Early stopping")
                break
