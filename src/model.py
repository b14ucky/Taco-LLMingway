import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as optim


class Attention(nn.Module):
    """
    Simple scaled dot-product self-attention mechanism with optional dropout.
    """

    def __init__(self, embed_dim: int, head_dim: int, dropout: float = 0.2) -> None:
        """
        Initializes the Attention module with linear projections for queries,
        keys, and values, and optional dropout on attention scores.

        Args:
            embed_dim: Dimensionality of the input embeddings.
            head_dim: Dimensionality of each attention head (output of q, k, v projections).
            dropout: Dropout probability applied to attention weights. Default is 0.2.
        """
        super().__init__()

        self.head_dim = head_dim
        self.q = nn.Linear(in_features=embed_dim, out_features=head_dim, bias=False)
        self.k = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=False)
        self.v = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """
        Computes scaled dot-product self-attention for the input tensor.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim).
            mask: Optional mask tensor of shape (batch_size, seq_len, seq_len)
                where 0 indicates positions to mask (ignored) and 1 indicates
                valid positions. Default is None.

        Returns:
            Tensor of shape (batch_size, seq_len, head_dim) containing
            the attended output after applying softmax and dropout.
        """
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        scores = q @ k.transpose(-2, -1) * self.head_dim**-0.5
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = self.softmax(scores)

        return self.dropout(attn_weights) @ v


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism that combines multiple single-head
    attentions and projects the concatenated output back to the embedding dimension.
    """

    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.2) -> None:
        """
        Initializes the MultiHeadAttention module.

        Args:
            embed_dim: Dimensionality of the input embeddings.
            n_heads: Number of attention heads. Must divide embed_dim evenly.
            dropout: Dropout probability applied to attention weights in each head. Default is 0.2.
        """
        super().__init__()

        assert embed_dim % n_heads == 0, "embed_dim should be divisible by n_heads"

        head_dim = embed_dim // n_heads
        self.attention = nn.ModuleList(
            modules=[Attention(embed_dim, head_dim, dropout) for _ in range(n_heads)]
        )
        self.linear = nn.Linear(in_features=embed_dim, out_features=embed_dim)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """
        Computes multi-head self-attention on the input tensor, applying the same mask to all heads.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim).
            mask: Optional mask tensor of shape (batch_size, seq_len, seq_len)
                where 0 indicates positions to mask (ignored) and 1 indicates
                valid positions. Default is None.

        Returns:
            Tensor of shape (batch_size, seq_len, embed_dim) containing
            the multi-head attended output after concatenation and linear projection.
        """
        x = torch.cat([attention(x, mask) for attention in self.attention], dim=-1)
        return self.linear(x)
