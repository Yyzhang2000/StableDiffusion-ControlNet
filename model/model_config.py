from dataclasses import dataclass


N_EMBED = 768


@dataclass
class CLIPConfig:
    n_vocab: int = 49408
    n_embed: int = N_EMBED
    n_token: int = 77

    # Attention
    n_head: int = 12

    # Clip layers
    n_layer: int = 12


# Diffusion
@dataclass
class DiffusionConfig:
    n_vocab: int = 49408
    n_embed: int = 320
    n_token: int = 77


@dataclass
class ModelConfig:

    clip_config = CLIPConfig
    diffusion_config = DiffusionConfig
