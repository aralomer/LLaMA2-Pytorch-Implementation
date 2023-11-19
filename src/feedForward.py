from args import ModelArgs
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        hidden_dim = 4 * args.DIM
        hidden_dim = int(2 * hidden_dim / 3)

        if args.FFn_DIM_MULTIPLIER is not None:
            hidden_dim = int(args.FFn_DIM_MULTIPLIER * hidden_dim)

        hidden_dim = args.MULTIPLE_OF * ((hidden_dim + args.MULTIPLE_OF - 1) // args.MULTIPLE_OF)

        self.w1 = nn.Linear(args.DIM, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.DIM, bias=False)
        self.w3 = nn.Linear(args.DIM, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        swish = F.silu(self.w1(x))
        x_V = self.w3(x)
        x = swish * x_V
        x = self.w2(x)

        return x
