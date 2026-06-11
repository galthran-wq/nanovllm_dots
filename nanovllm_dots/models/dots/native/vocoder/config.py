"""Native AudioVAEConfig: a dict-backed config mirroring the reference vocoder
config (a pydantic ConfigBase), which AudioVAE reads via BOTH attribute access
(``h.sample_rate``) and ``h.get("num_decoder_lookahead", 2)`` for undeclared keys."""
from __future__ import annotations

_DEFAULTS = {
    "sample_rate": 24000,
    "upsample_rates": [],
    "upsample_kernel_sizes": [],
    "upsample_initial_channel": 1536,
    "resblock": "1",
    "resblock_kernel_sizes": [],
    "resblock_dilation_sizes": [],
    "downsample_rates": [],
    "downsample_channels": [],
    "activation": "snakebeta",
    "snake_logscale": True,
    "latent_dim": 128,
    "causal": False,
    "mi_num_layers": 4,
    "causal_encoder": False,
    "use_bias_at_final": True,
    "use_tanh_at_final": True,
}


class AudioVAEConfig:
    def __init__(self, d: dict | None = None):
        self._d = {**_DEFAULTS, **(d or {})}

    @classmethod
    def from_dict(cls, d: dict) -> "AudioVAEConfig":
        return cls(d)

    def get(self, key, default=None):
        return self._d.get(key, default)

    def __getattr__(self, name):
        try:
            return self._d[name]
        except KeyError as e:
            raise AttributeError(name) from e


__all__ = ["AudioVAEConfig"]
