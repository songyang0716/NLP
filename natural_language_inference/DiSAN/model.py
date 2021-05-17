import torch
import torch.nn as nn


class DirectionalSelfAttention(nn.Module):
	"""
	Implement directional self-attention
	"""
    def __init__(self, embedding, embedding_size, dh, positional_mask="forward"):
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
    	self.W_w.bias.requires_grad_(False)

        self.tanh = nn.Tanh()
        self.softmax_layer = nn.Softmax(dim=1)

        self.c = torch.tensor([5])

    def forward(self, sentence_batch, sequence_length):
    	h = self.tanh(self.fc(self.emb(sentence_batch)))
    	print(h.size())



