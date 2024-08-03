import torch


class Residual(torch.nn.Module):
    def __init__(self, size, dropout) -> None:
        super(Residual, self).__init__()

        self.norm = torch.nn.LayerNorm(size)
        self.dropout = torch.nn.Dropout(p=dropout)

        def forward(self, x, sublayer):
            return x + self.dropout(sublayer(self.norm(x)))