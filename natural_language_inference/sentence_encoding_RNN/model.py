import torch
import torch.nn as nn


# Create RNN to encode hypothesis and premise
class Sentence_RNN(nn.Module):
    """
        Sentence RNN
    """
    def __init__(self,
                 hidden_size,
                 num_classes,
                 embeddings,
                 embedding_size,
                 num_layers):
        super(Sentence_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.emb = nn.Embedding.from_pretrained(embeddings=embeddings,
                                                freeze=True)
        self.rnn_hyp = nn.RNN(input_size=embedding_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              batch_first=True)
        self.rnn_prem = nn.RNN(input_size=embedding_size,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               batch_first=True)
        self.fc1 = nn.Linear(2 * hidden_size, 500)
        self.fc2 = nn.Linear(500, num_classes)
        self.activation = nn.ReLU()

    def forward(self, hyp_batch, premise_batch):
        """
            Extract the last hidden layer of each rnn
            Concatenate these two hiddens layers and then run a FC
        """
        hyp_embedding_layer = self.emb(hyp_batch)
        prem_embedding_layer = self.emb(premise_batch)

        hyp_out, hyp_hn = self.rnn_hyp(hyp_embedding_layer)
        prem_out, prem_hn = self.rnn_prem(prem_embedding_layer)

        hyp_hn = torch.squeeze(hyp_hn)
        prem_hn = torch.squeeze(prem_hn)

        combined_out = torch.cat((hyp_hn, prem_hn), dim=1)

        second_last_out = self.activation(self.fc1(combined_out))
        out = self.fc2(second_last_out)
        return out


class Sentence_LSTM(nn.Module):
    """
        Originally implemented in the paper
             "A large annotated corpus for learning natural language inference"
    """
    def __init__(self,
                 hidden_size,
                 num_classes,
                 embeddings,
                 embedding_size,
                 num_layers):
        super(Sentence_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.emb = nn.Embedding.from_pretrained(embeddings=embeddings,
                                                freeze=True)
        self.rnn_hyp = nn.RNN(input_size=embedding_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              batch_first=True)
        self.rnn_prem = nn.RNN(input_size=embedding_size,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               batch_first=True)
        self.fc1 = nn.Linear(2 * hidden_size, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 200)
        self.fc4 = nn.Linear(200, num_classes)
        # self.activation = nn.ReLU()

    def forward(self, hyp_batch, premise_batch):
        """
            Extract the last hidden layer of each rnn
            Concatenate these two hiddens layers and then run a FC
        """
        hyp_embedding_layer = self.emb(hyp_batch)
        prem_embedding_layer = self.emb(premise_batch)

        hyp_out, hyp_hn = self.rnn_hyp(hyp_embedding_layer)
        prem_out, prem_hn = self.rnn_prem(prem_embedding_layer)

        hyp_hn = torch.squeeze(hyp_hn)
        prem_hn = torch.squeeze(prem_hn)

        combined_out = torch.cat((hyp_hn, prem_hn), dim=1)

        second_last_out = self.activation(self.fc1(combined_out))
        out = self.fc2(second_last_out)
        return out

