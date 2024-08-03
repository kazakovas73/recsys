import torch
from .token import TokenEmbeddings
from .position import PositionalEmbeddings


class BertEmbeddings(torch.nn.Module):
    def __init__(self, vocab_size, embed_size, max_len, dropout=0.1) -> None:
        super().__init__()

        self.token = TokenEmbeddings(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbeddings(max_len=max_len, d_model=embed_size)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, seq):
        x = self.token(seq) + self.position(seq)
        return self.dropout(x)