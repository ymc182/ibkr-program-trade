import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import DataLoader, Dataset


class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(
            d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        if src_mask is None:
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            src_mask = nn.Transformer.generate_square_subsequent_mask(
                len(src)).to(device)
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TokenizedTextDataset(Dataset):
    """A Dataset class that holds tokenized text data."""

    def __init__(self, data, vocab):
        self.data = data
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        token_ids = [self.vocab[token] if token in self.vocab else self.vocab["<unk>"]
                     for token in self.data[idx]]
        return torch.tensor(token_ids, dtype=torch.long)


def collate_batch(batch):
    """A function to collate data into batches."""
    batch_size = len(batch)
    max_length = max(len(item) for item in batch)
    padded_batch = torch.zeros((max_length, batch_size), dtype=torch.long)
    for i, item in enumerate(batch):
        padded_batch[:len(item), i] = item
    return padded_batch


vocab = {"<unk>": 0, "<pad>": 1, "hello": 2, "world": 3,
         "!": 4, "goodbye": 5, "moon": 6, "sun": 7, "lol": 8, "how": 9, "are": 10, "you": 11, "doing": 12, "today": 13, "?": 14}
data = [["hello", "world", "!"], ["hello", "!"], ["goodbye",
                                                  "moon", "sun"], ["goodbye", "sun"], ["goodbye", "moon", "lol"], ["how", "are", "you", "doing", "today", "?"]]

dataset = TokenizedTextDataset(data, vocab)
print(dataset[0])
print(dataset[1])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_loader = DataLoader(dataset, batch_size=2, collate_fn=collate_batch)
ntokens = len(vocab)  # the size of vocabulary
emsize = 200  # embedding dimension
d_hid = 200  # dimension of the feedforward network model in ``nn.TransformerEncoder``
nlayers = 2  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
nhead = 2  # number of heads in ``nn.MultiheadAttention``
dropout = 0.2  # dropout probability
model = TransformerModel(ntokens, emsize, nhead, d_hid,
                         nlayers, dropout).to(device)

criterion = nn.CrossEntropyLoss()
lr = 1  # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
model.train()
for batch in data_loader:
    src = batch.to(device)
    tgt = src
    optimizer.zero_grad()
    output = model(src)
    loss = criterion(output.view(-1, ntokens), tgt.view(-1))
    loss.backward()
    optimizer.step()
    scheduler.step()
    print(loss.item())
# Test the model
model.eval()
word = "how are you".split()
src = torch.tensor([[vocab[token] for token in word]],
                   dtype=torch.long).to(device)
output = model(src)
print(output)

# convert to predictions
_, predicted = torch.max(output, 2)
predicted = predicted.cpu().numpy()
predicted_words = [list(vocab.keys())[list(vocab.values()).index(
    token)] for token in predicted[0]]

print(predicted_words)
