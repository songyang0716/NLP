import torch
import torch.nn as nn

# Create RNN to encode hypothesis and premise
class encoder(nn.Module):
    """
        Sentence RNN
    """

    def __init__(self,
                 hidden_size,
                 num_classes,
                 embeddings,
                 embedding_size,
                 num_layers):
        super(encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.emb = nn.Embedding.from_pretrained(embeddings=embeddings,
                                                freeze=True)
        self.lstm_prem = nn.LSTM(input_size=embedding_size,
                                 hidden_size=hidden_size,
                                 num_layers=num_layers,
                                 batch_first=True)
        self.fc1 = nn.Linear(2 * hidden_size, 500)
        self.fc2 = nn.Linear(500, num_classes)
        self.activation = nn.ReLU()

    def forward(self, prem_batch, premise_batch):
        """
            Extract the last hidden layer of the premise
        """
        hyp_embedding_layer = self.emb(prem_batch)
        prem_out, (prem_hn, prem_cn) = self.lstm_prem(prem_embedding_layer)
