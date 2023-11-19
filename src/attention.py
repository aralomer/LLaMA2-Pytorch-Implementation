import math
import torch
import torch.nn as nn
from args import ModelArgs
from utils import Utils
import torch.nn.functional as F


class SelfAttention(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        # Indicates the number of heads for the Key and Values
        self.n_kv_heads = args.N_HEADS if args.N_KV_HEADS is None else args.N_KV_HEADS

        # Indicates the number of heads for the Queries
        self.n_heads_q = args.N_HEADS

        # Indicates how many times the heads of Keys and Values should be repeated to match the head of Queries
        self.n_rep = self.n_heads_q // self.n_kv_heads

        # Indicates the dimension of each head
        self.head_dim = args.DIM // args.N_HEADS

        self.wq = nn.Linear(args.DIM, args.N_HEADS * self.head_dim, bias=True)
        self.wk = nn.Linear(args.DIM, self.n_kv_heads * self.head_dim, bias=True)
        self.wv = nn.Linear(args.DIM, self.n_kv_heads * self.head_dim, bias=True)
        self.wo = nn.Linear(args.N_HEADS * self.head_dim, args.DIM, bias=True)

        self.cache_k = torch.zeros((args.MAX_BATCH_SIZE, args.MAX_SEQ_LEN, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.MAX_BATCH_SIZE, args.MAX_SEQ_LEN, self.n_kv_heads, self.head_dim))

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        # (B, 1(Seq_Len), Dim)
        batch_size, seq_len, _ = x.shape

        # Apply the Wq, Wk and Wv matrices to queries, keys and values
        # (B, 1(Seq_Len), Dim) -> (B, 1(Seq_Len), H_Q * Head_Dim)
        xq = self.wq(x)

        # (B, 1(Seq_Len), Dim) -> (B, 1(Seq_Len), H_KV * Head_Dim)
        xk = self.wk(x)
        xv = self.wv(x)

        # (B, 1(Seq_Len), H_Q * Head_Dim) --> (B, 1(Seq_Len), H_Q, Head_Dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_heads_q, self.head_dim)

        # (B, 1(Seq_Len), H_KV * Head_Dim) --> (B, 1(Seq_Len), H_KV, Head_Dim)
        xv = xv.view(batch_size, seq_len, self.n_heads_q, self.head_dim)

        # Does not change the shape of the tensors
        xq = Utils.apply_rotary_embeddings(xq, freqs_complex, device=x.device)
        xk = Utils.apply_rotary_embeddings(xk, freqs_complex, device=x.device)

        # Replace the entry in the cache for this token
        self.cache_k[: batch_size, start_pos:start_pos + seq_len] = xk
        self.cache_v[: batch_size, start_pos:start_pos + seq_len] = xv

        # Retrieve all the cached keys and values so far
        # (B, Seq_Len_KV, H_KV, Head_Dim)
        keys = self.cache_k[: batch_size, 0: start_pos + seq_len]
        values = self.cache_v[: batch_size, 0: start_pos + seq_len]

        # Repeat the heads of the K and V to reach the number of heads of the queries
        keys = Utils.repeat_kv(keys, self.n_rep)
        values = Utils.repeat_kv(values, self.n_rep)

        # (B, 1(Seq_Len), H_Q, Head_Dim) --> (B, Seq_Len, 1(H_Q), Head_Dim)
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # (B, H_Q, 1, Head_Dim) @ (B, H, H_Q, 1, Head_Dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # (B, H_Q, 1, Head_Dim) @ (B, H_Q, Seq_Len_KV, HeadDim) --> (B, H_Q, 1, Head_Dim)
        output = torch.matmul(scores, values)

        # (B, H_Q, 1, Head_Dim) -> (B, 1, H_Q, Head_Dim)
        output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))
        return output
