import torch


class PositionalEmbeddings(torch.nn.Module):
    def __init__(self, max_len, d_model) -> None:
        super().__init__()

        self.pos_emb = torch.nn.Embedding(max_len, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        return self.pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)