import torch
import torch.nn as nn


class DirectionalSelfAttention(nn.Module):
    """
    Implement directional self-attention
    """
    def __init__(self, embedding, embedding_size, dh, device):
        super(DirectionalSelfAttention, self).__init__()
        self.emb = nn.Embedding.from_pretrained(
            embeddings=embedding, freeze=True
        )

        self.fc = nn.Linear(embedding_size, dh)
        self.W_1 = nn.Linear(dh, dh)
        self.W_2 = nn.Linear(dh, dh)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.xavier_uniform_(self.W_1.weight)
        nn.init.xavier_uniform_(self.W_2.weight)
        nn.init.constant_(self.W_1.bias, 0)
        nn.init.constant_(self.W_2.bias, 0)

        self.W_1.bias.requires_grad_(False)
        self.W_2.bias.requires_grad_(False)

        self.tanh = nn.Tanh()
        self.b_1 = nn.Parameter(torch.zeros(dh))
        self.c = nn.Parameter(torch.Tensor([5.0]), requires_grad=False)
        
        self.device = device

    def forward(self, sentence_batch, positional_mask="forward"):
        h = self.tanh(self.fc(self.emb(sentence_batch)))
        batch_size, seq_len, dh = h.size()[0], h.size()[1], h.size()[2]

        # The size of h is batch_size * setence_length * dh
        h = h.unsqueeze(2).expand(batch_size, seq_len, seq_len, dh)
        mask = self.return_positional_mask(batch_size, seq_len, dh, positional_mask)
        mask = mask.unsqueeze(-1).expand(batch_size, seq_len, seq_len, dh)
        l = self.c * self.tanh((torch.matmul(self.W_1, h) +
                                torch.matmul(self.W_2, h) + self.b_1) / self.c) + \
            mask
        return l

    def return_positional_mask(self,
                               batch_size,
                               seq_len,
                               dh,
                               positional_mask):
        mask = torch.ones(size=(seq_len, seq_len),
                          device=self.device)
        if positional_mask == 'forward':
            mask = torch.triu(mask, diagonal=1)
            mask[mask == 0] = float('-inf')
            mask[mask == 1] = 0
        elif positional_mask == 'backward':
            mask = torch.tril(mask, diagonal=-1)
            mask[mask == 0] = float('-inf')
            mask[mask == 1] = 0
        elif positional_mask == 'diagonal':
            mask = 0
            mask.fill_diagonal_(float('-inf'))
        else:
            raise NotImplementedError('only forward or backward mask is allowed!')
        return mask.unsqueeze(0).expand(batch_size, seq_len, seq_len)

