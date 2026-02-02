import torch
import torch.nn as nn
from torch import Tensor


class SinusoidalEmbeddings(nn.Module):
    """
    Generates fixed sinusoidal positional embeddings as introduced in
    "Attention Is All You Need" (Vaswani et al., 2017).
    """

    def __init__(self, context_len: int, embed_dim: int) -> None:
        """
        Initializes the sinusoidal positional embeddings.

        Args:
            context_len: Maximum sequence length (number of positions).
            embed_dim: Dimensionality of the embedding vector for each position.
        """
        super().__init__()

        position = torch.arange(context_len, dtype=torch.float32).unsqueeze(1)
        divisor = torch.tensor(10_000) ** (
            torch.arange(0, embed_dim, 2, dtype=torch.float32) / embed_dim
        )

        self._embeddings = torch.zeros(context_len, embed_dim, requires_grad=False)
        self._embeddings[:, 0::2] = torch.sin(position / divisor)
        self._embeddings[:, 1::2] = torch.cos(position / divisor)

    def forward(self, pos: Tensor) -> Tensor:
        """
        Retrieves positional embeddings for the given positions.

        Args:
            pos: Tensor of positions for which to return embeddings
                (e.g., shape [...], containing indices in [0, context_len)).

        Returns:
            A tensor of positional embeddings corresponding to the input positions.
        """
        return self._embeddings[pos]
