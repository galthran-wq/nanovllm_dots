"""nanovllm_dots — efficient batched inference engine for dots.tts-soar.

Forked from nano-vllm-voxcpm: reuses the model-agnostic runtime
(engine/ layers/ utils/) and adds a dots.tts model integration under
models/dots/.
"""

try:
    from nanovllm_dots._version import version as __version__
except Exception:
    __version__ = "0.0.0"

__all__ = ["__version__"]
