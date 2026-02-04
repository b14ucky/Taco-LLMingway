import torch
import torch.nn as nn
from torch import Tensor
from .embeddings import SinusoidalEmbeddings


class Attention(nn.Module):
    """
    Simple scaled dot-product self-attention mechanism with optional dropout.
    """

    def __init__(self, embed_dim: int, head_dim: int, dropout: float = 0.1) -> None:
        """
        Initializes the Attention module with linear projections for queries,
        keys, and values, and optional dropout on attention scores.

        Args:
            embed_dim: Dimensionality of the input embeddings.
            head_dim: Dimensionality of each attention head (output of q, k, v projections).
            dropout: Dropout probability applied to attention weights. Default is 0.1.
        """
        super().__init__()

        self.head_dim = head_dim
        self.q = nn.Linear(in_features=embed_dim, out_features=head_dim, bias=False)
        self.k = nn.Linear(in_features=embed_dim, out_features=head_dim, bias=False)
        self.v = nn.Linear(in_features=embed_dim, out_features=head_dim, bias=False)
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

    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.1) -> None:
        """
        Initializes the MultiHeadAttention module.

        Args:
            embed_dim: Dimensionality of the input embeddings.
            n_heads: Number of attention heads. Must divide embed_dim evenly.
            dropout: Dropout probability applied to attention weights in each head. Default is 0.1.
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


class Decoder(nn.Module):
    """
    Transformer-style decoder block with masked multi-head self-attention,
    a feed-forward network, residual connections, dropout, and layer normalization.
    """

    def __init__(
        self, embed_dim: int, n_heads: int, ffn_dim: int, dropout: float = 0.1
    ) -> None:
        """
        Initializes a single Transformer decoder block.

        Args:
            embed_dim: Dimensionality of the input and output embeddings.
            n_heads: Number of attention heads. Must divide embed_dim evenly.
            ff_dim: Hidden dimensionality of the feed-forward network.
            dropout: Dropout probability applied after attention and FFN. Default is 0.1.
        """
        super().__init__()

        assert embed_dim % n_heads == 0, "embed_dim should be divisible by n_heads"

        self.attention = MultiHeadAttention(embed_dim, n_heads, dropout)
        self.FFN = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=ffn_dim),
            nn.ReLU(),
            nn.Linear(in_features=ffn_dim, out_features=embed_dim),
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """
        Passes the input through self-attention and a feed-forward network
        with residual connections, dropout, and layer normalization.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim).
            mask: Optional causal or padding mask of shape
                (batch_size, seq_len, seq_len). Default is None.

        Returns:
            Tensor of shape (batch_size, seq_len, embed_dim) containing
            the updated hidden representations.
        """
        attn_out = self.dropout(self.attention(self.norm(x), mask))
        x = x + attn_out

        ffn_out = self.dropout(self.FFN(self.norm(x)))
        x = x + ffn_out

        return x


class GPT(nn.Module):
    """
    GPT-style causal language model using sinusoidal positional embeddings
    and a stack of Transformer decoder blocks.

    The model predicts the next token in a sequence in an autoregressive manner.
    """

    def __init__(
        self,
        embed_dim: int,
        vocab_size: int,
        context_len: int,
        n_heads: int,
        ffn_dim: int,
        n_blocks: int,
        dropout: float = 0.1,
    ) -> None:
        """
        Initializes the GPT model.

        Args:
            embed_dim: Dimensionality of token and hidden embeddings.
            vocab_size: Size of the vocabulary.
            context_len: Maximum sequence length the model can process.
            n_heads: Number of attention heads in each decoder block.
            ffn_dim: Hidden dimensionality of the feed-forward network in each decoder block.
            n_blocks: Number of stacked decoder blocks.
            dropout: Dropout probability used throughout the model. Default is 0.1.
        """
        super().__init__()

        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.pos_embeddings = SinusoidalEmbeddings(context_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(in_features=embed_dim, out_features=vocab_size)
        self.blocks = nn.ModuleList(
            [Decoder(embed_dim, n_heads, ffn_dim, dropout) for _ in range(n_blocks)]
        )

        self.context_len = context_len

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the GPT model.

        The input token sequence is:
        1) Embedded into vector representations,
        2) Enriched with positional embeddings,
        3) Passed through stacked causal decoder blocks with a causal mask,
        4) Normalized and projected to vocabulary logits.

        Args:
            x: Input tensor of token indices with shape (batch_size, seq_len).

        Returns:
            Tensor of shape (batch_size, seq_len, vocab_size) containing
            unnormalized logits for next-token prediction at each position.
        """
        _, L = x.size()
        if L > self.context_len:
            x = x[:, : self.context_len]
            L = self.context_len

        positions = torch.arange(L).to(x.device)
        x = self.embeddings(x) + self.pos_embeddings(positions).unsqueeze(0)

        mask = torch.tril(torch.ones(L, L, device=x.device)).unsqueeze(0)

        for block in self.blocks:
            x = block(x, mask)

        return self.linear(self.norm(x))
