import torch


class FFN(torch.nn.Module):
    def __init__(self, d_model, d_ffn, dropout=0.1) -> None:
        super(FFN, self).__init__()

        self.w1 = torch.nn.Linear(d_model, d_ffn)
        self.w2 = torch.nn.Linear(d_ffn, d_model)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.act = torch.nn.GELU()

    def forward(self, x):
        return self.w2(self.dropout(self.act(self.w1(x))))
