import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
import torchvision.transforms as T
import torch.nn.functional as F


class SkipGramNeg(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super(SkipGramNeg, self).__init__()
        self.input_emb = nn.Embedding(vocab_size, emb_dim)
        self.output_emb = nn.Embedding(vocab_size, emb_dim)
        self.log_sigmoid = nn.LogSigmoid()

          # Xavier init
        self.input_emb.weight.data = nn.init.xavier_uniform_(self.input_emb.weight.data)
        self.output_emb.weight.data = nn.init.xavier_uniform_(self.output_emb.weight.data)





    def forward(self, target_input, context, neg):
        """
        :param target_input: [batch_size]
        :param context: [batch_size]
        :param neg: [batch_size, neg_size]
        :return:
        """
        # u,v: [batch_size, emb_dim]
        v = self.input_emb(target_input)
        u = self.output_emb(context)

        positive_val = self.log_sigmoid(torch.sum(u * v, dim=1)).squeeze()

		# u_hat: [batch_size, neg_size, emb_dim]
		# for each positive sample, we will randomly pick k negative samples
        u_hat = self.output_emb(neg)

        loss = positive_val + neg_val
        return -loss.mean()


    def predict(self, inputs):
        return self.input_emb(inputs)


