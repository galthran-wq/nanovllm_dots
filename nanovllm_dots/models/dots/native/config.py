"""Config parsing for the native dots model (from the checkpoint's config.json)."""
import json
import os


class TransformerConfig:
    """Thin wrapper over a config.json sub-block (``DiT`` / ``PatchEncoder``).

    The native DiT/encoder only need ``.to_dict()`` (passed as **kwargs into the
    layers, which pick what they use) plus ``.hidden_size`` / ``.num_layers``.
    """

    def __init__(self, d: dict):
        self._d = dict(d)

    def to_dict(self) -> dict:
        # drop None values to match the reference ConfigBase.to_dict (exclude_none):
        # e.g. a null rotary_theta must fall back to the layer default, not be passed.
        return {k: v for k, v in self._d.items() if v is not None}

    def get(self, key, default=None):
        return self._d.get(key, default)

    def __getattr__(self, name):
        try:
            return self._d[name]
        except KeyError as e:
            raise AttributeError(name) from e


class PatchEncoderModelConfig:
    """Mirrors what VAESemanticEncoder reads from the core config: ``.patch_size``
    (int) and ``.PatchEncoder`` (a TransformerConfig with attr access + ``.get()``)."""

    def __init__(self, raw: dict):
        self.patch_size = int(raw["patch_size"])
        self.PatchEncoder = TransformerConfig(raw["PatchEncoder"])


def load_config(model_dir: str) -> dict:
    """The raw config.json for a model directory."""
    with open(os.path.join(model_dir, "config.json")) as f:
        return json.load(f)


def dit_config(model_dir: str) -> TransformerConfig:
    return TransformerConfig(load_config(model_dir)["DiT"])


def patch_encoder_config(model_dir: str) -> TransformerConfig:
    return TransformerConfig(load_config(model_dir)["PatchEncoder"])
