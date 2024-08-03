import torch
from .embeddings import BertEmbeddings
from .transformer import TransformerBlock



class Bert4Rec(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        max_len = args.max_len
        n_layers = args.num_layers
        heads = args.num_heads
        vocab_size = args.vocab_size
        hidden = args.hidden
        dropout = args.dropout

        self.embeddings = BertEmbeddings(
            vocab_size=vocab_size,
            embed_size=hidden,
            max_len=max_len,
            dropout=dropout)
        
        self.blocks = torch.nn.ModuleList(
            [TransformerBlock(hidden=hidden, 
                              attn_heads=heads, 
                              ffn_hidden=hidden*4, 
                              dropout=dropout) for _ in range(n_layers)]
        )

    def forward(self, x):
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        x = self.embeddings(x)

        for block in self.blocks:
            x = block.forward(x, mask)
        
        return x

    def init_weights(self):
        pass
    