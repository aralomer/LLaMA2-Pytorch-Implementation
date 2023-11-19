from dataclasses import dataclass
import torch


@dataclass
class Utils():

    @staticmethod
    def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
        # (B, Seq_Len, H, Head_Dim) -> (B, Seq_Len, H, Head_Dim / 2)
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        # (Seq_Len, Head_Dim / 2) -> (1, Seq_Len, 1, Head_Dim / 2)
        freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
        # (B, Seq_Len, H, Head_Dim / 2) -> (1, Seq_Len, 1, Head_Dim / 2) = (B, Seq_Len, H, Head_Dim / 2)
        x_rotated = x_complex * freqs_complex
        # (B, Seq_Len, H, Head_Dim / 2) -> (B, Seq_Len, H, Head_Dim / 2, 2)
        x_out = torch.view_as_real(x_rotated)
        # (B, Seq_Len, H, Head_Dim / 2, 2) -> (B, Seq_Len, H, Head_Dim)
        x_out = x_out.reshape(*x.shape)
        return x_out.type_as(x).to(device)

    @staticmethod
    def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:

        batch_size, seq_len, n_kv_heads, head_dim = x.shape
        output = None
        if n_rep == 1:
            output = x
        else:
            # (B, Seq_Len, N_KV_HEADS, 1, Head_Dim)
            output = (x[:, :, :, None, :]
                      .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
                      .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim))

        return output

    @staticmethod
    def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
        # As written in the paper, the dimension of the embedding must be even
        assert head_dim % 2 == 0, 'Dimension must be divisible by 2'

        # Build the theta parameters
        # According to the formula theta_i = 10000 ^ (-2(i-1/dim) for i = [1, 2, ... dim / 2]
        # Shape: (Head / 2)
        theta_numerator = torch.arange(0, head_dim, 2).float()
        # Shape: (Head / 2)
        theta = 1.0 / theta ** (theta_numerator / head_dim).to(device)
        # Construct the position (the "m" parameter)
        # Shape: (Seq_Len)
        m = torch.arange(seq_len, device=device)
        # Multiply each theta by each position using the outer product
        # Shape : Seq_Len outer_product * (Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
        freqs = torch.outer(m, theta).float()
        # We can compute complex numbers in the polar from c = R * exp(i * m * theta), where R = 1 as follows:
        # (Seq_Len, Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
        freqs_complex = torch.polar(torch.ones_like(freqs), freqs)

        return freqs_complex
