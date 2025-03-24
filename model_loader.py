import model_converter

from model import CLIP, Diffusion, VAE_Decoder, VAE_Encoder
from model.model_config import ModelConfig


def preload_models_from_standard_weights(ckpt_path, device):
    state_dict = model_converter.load_from_standard_weights(ckpt_path, device)

    encoder = VAE_Encoder().to(device)
    encoder.load_state_dict(state_dict["encoder"], strict=True)

    decoder = VAE_Decoder().to(device)
    decoder.load_state_dict(state_dict["decoder"], strict=True)

    diffusion = Diffusion(ModelConfig.diffusion_config).to(device)
    diffusion.load_state_dict(state_dict["diffusion"], strict=True)

    clip = CLIP(ModelConfig.clip_config).to(device)
    clip.load_state_dict(state_dict["clip"], strict=True)

    return {
        "clip": clip,
        "encoder": encoder,
        "decoder": decoder,
        "diffusion": diffusion,
    }
