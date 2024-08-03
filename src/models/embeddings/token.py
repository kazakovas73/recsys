import torch

class TokenEmbeddings(torch.nn.Module):
    def __init__(self, vocab_size, embed_size=128) -> None:
        super().__init__(vocab_size, embed_size, padding_idx=0)

