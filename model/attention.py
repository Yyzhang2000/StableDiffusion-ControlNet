import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)

        self.n_heads = n_heads
        assert (
            d_embed % n_heads == 0
        ), "Embedding dimension must be divisible by number of heads."
        self.d_head = d_embed // n_heads

    def forward(self, x, causal_mask=False):
        B, T, C = x.size()

        q, k, v = map(
            lambda t: t.view(B, T, self.n_heads, self.d_head).transpose(1, 2),
            self.in_proj(x).chunk(3, dim=-1),
        )

        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        if causal_mask:
            mask = torch.ones_like(score).triu(1)
            score = score.masked_fill(mask == 1, float("-inf"))

        score = F.softmax(score, dim=-1)
        attn = torch.matmul(score, v).transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(attn)


class CrossAttention(nn.Module):
    def __init__(
        self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True
    ):
        super().__init__()

        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)

        self.n_heads = n_heads
        assert (
            d_embed % n_heads == 0
        ), "Embedding dimension must be divisible by number of heads."
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """Cross Attention Between Image and Text

        Args:
            x (torch.Tensor): Image features
            y (torch.Tensor): Text features
        """

        B, S, C = x.size()

        interim_shape = (B, -1, self.n_heads, self.d_head)

        q = self.q_proj(x).view(*interim_shape).transpose(1, 2)

        k, v = map(
            lambda t: t.view(B, -1, self.n_heads, self.d_head).transpose(1, 2),
            (self.k_proj(y), self.v_proj(y)),
        )

        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        score = F.softmax(score, dim=-1)

        attn = torch.matmul(score, v).transpose(1, 2).contiguous().view(B, S, C)

        return self.out_proj(attn)
