import torch
import torch.nn as nn
from args import ModelArgs
from layerNorm import RMSNorm
from attention import SelfAttention
from feedForward import FeedForward

class EncoderBlock(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.N_HEADS
        self.dim = args.DIM
        self.head_dim = args.DIM // args.N_HEADS

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        # Normalization before the self attention
        self.attention_norm = RMSNorm(args.DIM, eps=args.NORM_EPS)
        # Normalization before the feed forward block
        self.ffn_form = RMSNorm(args.DIM, eps=args.NORM_EPS)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) --> (B, Seq_Len, Dim)
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)
        out = h + self.feed_forward(self.ffn_form(h))
        return out
