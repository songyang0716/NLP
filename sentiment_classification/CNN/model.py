import torch
import torch.nn as nn
# import torchvision.transforms as T
import torch.nn.functional as F
class CNN(nn.Module):
    """
        Implementation of CNN for classification
    """
    def __init__(self, embeddings, input_dim, window_dim, filter_dim, output_dim, max_len, dropout):
        super(CNN, self).__init__()

        """
        Arguments
        ---------
        embeddings : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table
        dropout: Probability of retaining an activation node during dropout operation
        --------

        """

        self.max_len = max_len
        self.input_dim = input_dim
        self.window_dim = window_dim
        self.filter_dim = filter_dim
        self.output_dim = output_dim

        # Initial the embedding with glove
        # padding_idx means if the input is with index 0, then padding with zero vectors
        # https://stackoverflow.com/questions/61172400/what-does-padding-idx-do-in-nn-embeddings

        self.emb = nn.Embedding(embeddings.size(0), embeddings.size(1), padding_idx=0).requires_grad_(False)
        self.emb.weight = nn.Parameter(embeddings)

        self.cnn = nn.Conv2d(in_channels=1,
                             out_channels=filter_dim,
                             kernel_size=(window_dim, embeddings.size(1)),
                             stride=1,
                             padding=0)

        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(self.filter_dim, self.output_dim)

    def forward(self, sen_batch, sen_lengths):
        """
        :param sen_batch: shape: (batch, sen_length), tensor for sentence sequence
        :param sen_lengths:
        :return:
        """
        ''' Embedding Layer | Padding | Sequence_length 40'''

        sen_batch = self.emb(sen_batch) # sen_batch.size() = (batch_size, num_seq, embedding_length)
        batch_size = len(sen_batch)
        sen_batch = sen_batch.unsqueeze(1) # sen_batch.size() = (batch_size, 1, num_seq, embedding_length)
        conv_out = self.cnn(sen_batch) # conv_out.size() =  (batch_size, filter_dim, ?, 1)
        conv_out = F.relu(conv_out.squeeze(3)) # conv_out.size() = (batch_size, filter_dim, dim)
        max_out = F.max_pool1d(conv_out, kernel_size=conv_out.size()[2]).squeeze(2) # max_out.size() = (batch_size, filter_dim)
        fc_in = self.dropout(max_out)
        out = self.output(fc_in)
        return out 

