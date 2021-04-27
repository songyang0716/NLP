import torch
import torch.nn as nn


class encoder(nn.Module):
    """
        Encoding premise using biLSTM
    """

    def __init__(self,
                 hidden_size,
                 embeddings,
                 embedding_size,
                 num_layers,
                 p):
        super(encoder, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p)
        self.emb = nn.Embedding.from_pretrained(embeddings=embeddings,
                                                freeze=True)
        self.bilstm_layer = nn.LSTM(input_size=embedding_size,
                                    hidden_size=hidden_size,
                                    num_layers=num_layers,
                                    batch_first=True,
                                    dropout=p,
                                    bidirectional=True)

    def forward(self, prem_batch, sequence_length):
        """
            Feed word embedding batch into LSTM
        """
        prem_embedding_layer = self.dropout(self.emb(prem_batch))
        packed_embeddings = \
            nn.utils.rnn.pack_padded_sequence(prem_embedding_layer,
                                              sequence_length.cpu(),
                                              batch_first=True,
                                              enforce_sorted=False)
        prem_out, (prem_hn, prem_cn) = self.bilstm_layer(packed_embeddings)
        print("prem_out shape before packed", prem_out.size())

        prem_out, _ = nn.utils.rnn.pad_packed_sequence(prem_out,
                                                       batch_first=True)
        print("prem_out shape", prem_out.size())

        return prem_out, prem_hn, prem_cn


class inner_attention(nn.Module):
    """
        Perform inner attention over the premise and hypothesis bilstm output
        And return the final sentence vector
    """

    def __init__(self, encoder):
        super(inner_attention, self).__init__()
        self.encoder = encoder
        hidden_size = self.encoder.hidden_size
        self.WY = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.WH = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.W = nn.Linear(2 * hidden_size, 1)
        self.tanh = nn.Tanh()
        self.softmax_layer = nn.Softmax(dim=1)

    def forward(self, prem_batch, hyp_batch, prem_length, hyp_length):
        """
            Perform inner attention
        """
        prem_bilstm_output, _, _ = self.encoder(prem_batch, prem_length)
        hyp_bilstm_output, _, _ = self.encoder(hyp_batch, hyp_length)
        # shape is like batch * sequence * embedding
        prem_mean_vec = torch.mean(prem_bilstm_output, dim=1, keepdim=True)
        hyp_mean_vec = torch.mean(hyp_bilstm_output, dim=1, keepdim=True)
        prem_mean_vec = prem_mean_vec.permute(0, 2, 1)
        hyp_mean_vec = hyp_mean_vec.permute(0, 2, 1)

        M_premise = \
            self.WY(prem_bilstm_output) + \
            self.WH(torch.matmul(prem_mean_vec,
                                 torch.ones([1,
                                             prem_bilstm_output.shape[1]])).permute(0,2,1))
        # Size of M is like  batch * sequence * embedding
        M_premise = self.tanh(M_premise)
        # Size of weights premise is like batch * sequence
        weights_premise = self.softmax_layer(self.W(M_premise).squeeze(2))
        weighted_premise = torch.bmm(M_premise.permute(0, 2, 1), weights_premise.unsqueeze(2))

        M_hyp = \
            self.WY(hyp_bilstm_output) + \
            self.WH(torch.matmul(hyp_mean_vec,
                                 torch.ones([1,
                                             hyp_bilstm_output.shape[1]])).permute(0,2,1))
        # Size of M is like  batch * sequence * embedding
        M_hyp = self.tanh(M_hyp)
        # Size of weights premise is like batch * sequence
        weights_hyp = self.softmax_layer(self.W(M_hyp).squeeze(2))
        weighted_hyp = torch.bmm(M_hyp.permute(0, 2, 1), weights_hyp.unsqueeze(2))

        return weighted_hyp.squeeze(2) + weighted_premise.squeeze(2)


