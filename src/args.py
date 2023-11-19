from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelArgs:
    DIM: int = 4096
    N_LAYERS: int = 32
    # Number of heads for the queries
    N_HEADS: int = 32
    # Number of heads for the K(key) and V(value)
    N_KV_HEADS: Optional[int] = None
    # This will be set when we load the tokenizer
    VOCAB_SIZE: int = -1
    MULTIPLE_OF: int = 256
    FFn_DIM_MULTIPLIER: Optional[float] = None
    NORM_EPS: float = 1e-5

    MAX_BATCH_SIZE: int = 32
    MAX_SEQ_LEN: int = 2048

    DEVICE: str = None

