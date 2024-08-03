import torch
from .utils import FFN, Residual


class TransformerBlock(torch.nn.Module):
    def __init__(self, hidden, attn_heads, ffn_hidden, dropout) -> None:
        super().__init__()

        self.attention = torch.nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=attn_heads,
            dropout=dropout
        )

        self.ffn = FFN(d_model=hidden, d_ffn=ffn_hidden, dropout=dropout)
        self.inp_residual = Residual(size=hidden, dropout=dropout)
        self.out_residual = Residual(size=hidden, dropout=dropout)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.inp_residual(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.out_residual(x, self.ffn)
        return self.dropout(x)
