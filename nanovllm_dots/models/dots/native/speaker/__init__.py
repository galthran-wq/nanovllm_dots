"""Native CAM++ speaker x-vector encoder (ported from dots_tts/modules/speaker;
pure torch/torchaudio). Validated native==reference."""
from .encoder import SpeakerXVectorFeatures

__all__ = ["SpeakerXVectorFeatures"]
