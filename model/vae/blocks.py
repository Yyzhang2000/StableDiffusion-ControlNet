import torch
import torch.nn as nn
import torch.nn.functional as F

from ..attention import SelfAttention


class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x):
        h = x

        x = self.groupnorm(x)

        B, C, H, W = x.size()

        x = x.view(B, C, H * W).transpose(-1, -2)

        x = self.attention(x)

        x = x.view((B, C, H, W))

        x = h + x

        return x


class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        h = x

        x = F.silu(self.groupnorm_1(x))
        x = self.conv_1(x)

        x = F.silu(self.groupnorm_2(x))
        x = self.conv_2(x)

        return x + self.residual_layer(h)
