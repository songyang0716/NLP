import torch
import torch.nn as nn


class encoder1(nn.Module):
    """
        Encoding premise using LSTM
    """

    def __init__(self,
                 hidden_size,
                 embeddings,
                 embedding_size,
                 num_layers,
                 p):
        super(encoder1, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p)
        self.emb = nn.Embedding.from_pretrained(embeddings=embeddings,
                                                freeze=True)
        self.lstm_prem = nn.LSTM(input_size=embedding_size,
                                 hidden_size=hidden_size,
                                 num_layers=num_layers,
                                 batch_first=True,
                                 dropout=p)

    def forward(self, prem_batch, sequence_length):
        """
            Feed premise batch into LSTM
            Extract the last hidden layer of the premise
        """
        prem_embedding_layer = self.dropout(self.emb(prem_batch))
        packed_embeddings = \
            nn.utils.rnn.pack_padded_sequence(prem_embedding_layer,
                                              sequence_length.cpu(),
                                              batch_first=True,
                                              enforce_sorted=False)
        prem_out, (prem_hn, prem_cn) = self.lstm_prem(packed_embeddings)
        prem_out, _ = nn.utils.rnn.pad_packed_sequence(prem_out, batch_first=True)

        return prem_out, prem_hn, prem_cn


class encoder2(nn.Module):
    """
    Input last hidden state from encoder1
    Then encode hypothesis
    """

    def __init__(self,
                 hidden_size,
                 embeddings,
                 embedding_size,
                 num_layers,
                 p):
        super(encoder2, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p)
        self.emb = nn.Embedding.from_pretrained(embeddings=embeddings,
                                                freeze=True)
        self.lstm_hyp = nn.LSTM(input_size=embedding_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                batch_first=True,
                                dropout=p)

    def forward(self, hyp_batch, h0, c0, sequence_length):
        """
            Feed hypothesis batch into LSTM
            The first hidden state is from encoder1
        """
        hyp_embedding_layer = self.dropout(self.emb(hyp_batch))
        packed_embeddings = \
            nn.utils.rnn.pack_padded_sequence(hyp_embedding_layer,
                                              sequence_length.cpu(),
                                              batch_first=True,
                                              enforce_sorted=False)
        hyp_out, (hyp_hn, hyp_cn) = self.lstm_hyp(packed_embeddings, (h0, c0))

        return hyp_out, hyp_hn, hyp_cn


class attention(nn.Module):
    """
    Generate weighted average of premise vectors
    """
    def __init__(self, encoder1, encoder2, hidden_size):
        super(attention, self).__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.softmax_layer = nn.Softmax(dim=1)
        self.activation = nn.ReLU()
        self.fc1 = nn.Linear(2 * hidden_size, 100)
        self.fc2 = nn.Linear(100, 3)
        self.activation = nn.ReLU()
        self.fc1.weight.data = nn.init.xavier_uniform_(self.fc1.weight.data)
        self.fc2.weight.data = nn.init.xavier_uniform_(self.fc2.weight.data)

    def forward(self, prem_batch, hyp_batch, prem_length, hyp_length):
        prem_hiddens, prem_hn, prem_cn = self.encoder1(prem_batch, prem_length)
        _, hyp_hn, hyp_cn = self.encoder2(hyp_batch, prem_hn, prem_cn, hyp_length)


        # Multiple each premise hidden layer by last hidden layer
        # Apply softmax to return the weights
        hyp_hn = hyp_hn.permute(1, 2, 0)
        attention_weights = torch.squeeze(torch.bmm(prem_hiddens, hyp_hn), 2)
        attention_weights = self.softmax_layer(attention_weights).unsqueeze(dim=2)

        prem_hiddens = prem_hiddens.permute(0, 2, 1)
        weighted_prem_hiddens = torch.bmm(prem_hiddens, attention_weights)

        combined_out = torch.cat((hyp_hn, weighted_prem_hiddens), dim=1).squeeze(2)
        # print("combined_out", combined_out.size())
        second_last_out = self.activation(self.fc1(combined_out))
        # print("second_last_out", second_last_out.size())
        out = self.fc2(second_last_out)
        return out