import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import SelfAttention
from model_config import ModelConfig


class CLIPEmbedding(nn.Module):
    def __init__(self, config: ModelConfig.clip_config):
        super().__init__()

        self.token_embedding = nn.Embedding(config.n_vocab, config.n_embed)

        self.position_embedding = nn.Parameter(
            torch.zeros((config.n_token, config.n_embed))
        )

    def forward(self, tokens):
        x = self.token_embedding(tokens)
        x = x + self.position_embedding

        return x


class CLIPLayer(nn.Module):
    def __init__(self, config: ModelConfig.clip_config):
        super().__init__()

        self.layernorm_1 = nn.LayerNorm(config.n_embed)

        self.attention = SelfAttention(config.n_head, config.n_embed)

        self.layernorm_2 = nn.LayerNorm(config.n_embed)

        self.linear_1 = nn.Linear(config.n_embed, config.n_embed)
        self.linear_2 = nn.Linear(config.n_embed, config.n_embed)

    def forward(self, x):
        h = x

        x = self.layernorm_1(x)
        x = self.attention(x, causal_mask=True)
        x = x + h

        h = x

        x = self.layernorm_2(x)
        x *= torch.sigmoid(1.702 * x)  # Quick implementation of GELU

        x = self.linear_1(x)

        x = h + x
        return x


class CLIP(nn.Module):
    def __init__(self, config: ModelConfig.clip_config):
        super().__init__()

        self.embedding = CLIPEmbedding(config)
        self.layers = nn.ModuleList([CLIPLayer(config) for _ in range(config.n_token)])

        self.layernorm = nn.LayerNorm(config.n_embed)

    def forward(self, tokens):
        tokens = tokens.type(torch.long)

        x = self.embedding(tokens)

        for layer in self.layers:
            x = layer(x)

        return self.layernorm(x)
