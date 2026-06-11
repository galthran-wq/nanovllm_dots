"""Native BigVGAN AudioVAE vocoder (copied verbatim from the reference
dots_tts/modules/vocoder, which depends only on torch/numpy + the backbone
Conv1d/ConvTranspose1d). Validated native==reference."""
from .bigvgan import AudioVAE
from .config import AudioVAEConfig

__all__ = ["AudioVAE", "AudioVAEConfig"]
