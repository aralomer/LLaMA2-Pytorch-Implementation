import torch
import torch.nn as nn
from args import ModelArgs
from encoder import EncoderBlock
from layerNorm import RMSNorm
from utils import Utils


class Transformer(nn.Module):

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        assert args.VOCAB_SIZE != -1, 'Vocab size must be set!'

        self.args = args
        self.vocab_size = args.VOCAB_SIZE
        self.n_layers = args.N_LAYERS
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.DIM)

        self.layers = nn.ModuleList()

        for _ in range(args.N_LAYERS):
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args.DIM, eps=args.NORM_EPS)
        self.output = nn.Linear(args.DIM, self.vocab_size, bias=False)

        self.freqs_complex = Utils.precompute_theta_pos_frequencies(self.args.DIM // self.args.N_HEADS,
                                                                    self.args.MAX_SEQ_LEN * 2, device=self.args.DEVICE)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        # (B, Seq_Len)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, 'Only one token at a time can be processed'

        # (B, Seq_Len) -> (B, Seq_Len, Dim)
        h = self.tok_embeddings(tokens)

        # Retrieve the pairs (m, theta) corresponding to the position [start_pos, start_pos + seq_len]

        freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]

        # Consecutively apple all the encoder layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)

        output = self.output(h).float()
        return output
