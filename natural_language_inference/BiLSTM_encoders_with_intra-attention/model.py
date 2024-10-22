import torch
import torch.nn as nn


class encoder(nn.Module):
    """
    Encoding premise using biLSTM
    """

    def __init__(self, hidden_size, embeddings, embedding_size, num_layers, p):
        super(encoder, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p)
        self.emb = nn.Embedding.from_pretrained(
            embeddings=embeddings, freeze=True
        )
        self.bilstm_layer = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=p,
            bidirectional=True,
        )

    def forward(self, sentence_batch, sequence_length):
        """
        Feed word embedding batch into LSTM
        """
        sentence_embedding_layer = self.dropout(self.emb(sentence_batch))
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(
            sentence_embedding_layer,
            sequence_length.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        out, (hn, cn) = self.bilstm_layer(packed_embeddings)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        return out, hn, cn


class inner_attention(nn.Module):
    """
    Perform inner attention over the premise and hypothesis bilstm output
    And return the final sentence vector
    """

    def __init__(self, encoder, p=0.2):
        super(inner_attention, self).__init__()
        self.encoder = encoder
        hidden_size = self.encoder.hidden_size
        self.WY = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.WH = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.W = nn.Linear(2 * hidden_size, 1)
        self.fc1 = nn.Linear(8 * hidden_size, 300)
        self.fc2 = nn.Linear(300, 300)
        self.fc3 = nn.Linear(300, 3)
        self.tanh = nn.Tanh()
        self.softmax_layer = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p)

        self.WY.weight.data = nn.init.xavier_uniform_(self.WY.weight.data)
        self.WH.weight.data = nn.init.xavier_uniform_(self.WH.weight.data)
        self.W.weight.data = nn.init.xavier_uniform_(self.W.weight.data)

        self.fc1.weight.data = nn.init.xavier_uniform_(self.fc1.weight.data)
        self.fc2.weight.data = nn.init.xavier_uniform_(self.fc2.weight.data)
        self.fc3.weight.data = nn.init.xavier_uniform_(self.fc3.weight.data)

        self.layernorm = nn.LayerNorm(normalized_shape=300)

    def forward(self, prem_batch, hyp_batch, prem_length, hyp_length):
        """
        Perform inner attention on premise and hyp batch

        Arguments:
            prem_batch:  premise sentence indexes
            hyp_batch:   hypothesis sentence indexes
            prem_length: premise sentence length
            hyp_length:  hypothesis sentence length
        Returns:
            inner attentioned sentences
        """
        prem_bilstm_output, _, _ = self.encoder(prem_batch, prem_length)
        hyp_bilstm_output, _, _ = self.encoder(hyp_batch, hyp_length)
        # shape is like batch * sequence * embedding
        prem_mean_vec = torch.mean(prem_bilstm_output, dim=1, keepdim=True)
        hyp_mean_vec = torch.mean(hyp_bilstm_output, dim=1, keepdim=True)
        prem_mean_vec = prem_mean_vec.permute(0, 2, 1)
        hyp_mean_vec = hyp_mean_vec.permute(0, 2, 1)

        M_premise = self.WY(prem_bilstm_output) + self.WH(
            torch.matmul(
                prem_mean_vec,
                torch.ones([1, prem_bilstm_output.shape[1]], device="cuda"),
            ).permute(0, 2, 1)
        )
        # Size of M is like  batch * sequence * embedding
        M_premise = self.tanh(M_premise)
        # Size of weights premise is like batch * sequence
        weights_premise = self.softmax_layer(self.W(M_premise).squeeze(2))
        weighted_premise = torch.bmm(
            M_premise.permute(0, 2, 1), weights_premise.unsqueeze(2)
        )
        weighted_premise = weighted_premise.squeeze(2)

        M_hyp = self.WY(hyp_bilstm_output) + self.WH(
            torch.matmul(
                hyp_mean_vec,
                torch.ones([1, hyp_bilstm_output.shape[1]], device="cuda"),
            ).permute(0, 2, 1)
        )
        # Size of M is like  batch * sequence * embedding
        M_hyp = self.tanh(M_hyp)
        # Size of weights premise is like batch * sequence
        weights_hyp = self.softmax_layer(self.W(M_hyp).squeeze(2))
        weighted_hyp = torch.bmm(
            M_hyp.permute(0, 2, 1), weights_hyp.unsqueeze(2)
        )
        weighted_hyp = weighted_hyp.squeeze(2)

        sentence_difference = weighted_premise - weighted_hyp
        sentence_multiplication = weighted_premise * weighted_hyp

        sentence_matching = torch.cat(
            (
                weighted_premise,
                sentence_multiplication,
                sentence_difference,
                weighted_hyp,
            ),
            dim=1,
        )
        fc1_layer = self.fc1(sentence_matching)
        fc1_layer = self.tanh(fc1_layer)
        fc1_layer = self.layernorm(fc1_layer)
        fc1_layer = self.dropout(fc1_layer)

        fc2_layer = self.fc2(fc1_layer)
        fc2_layer = self.tanh(fc2_layer)
        fc2_layer = self.layernorm(fc2_layer)
        fc2_layer = self.dropout(fc2_layer)

        out = self.fc3(fc2_layer)
        return out
