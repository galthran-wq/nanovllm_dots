"""Weight loading for the native dots components.

The native modules mirror the reference structure exactly, so loading is a direct
prefix-stripped ``load_state_dict`` (no fused-projection remapping like the LLM
loader needs). Non-persistent buffers (rotary ``inv_freq``) are recomputed at
construction, so they show up as "missing" and are ignored.
"""
import os

import torch
from safetensors import safe_open

from .config import PatchEncoderModelConfig, dit_config, load_config
from .dit import DiT
from .patch_encoder import VAESemanticEncoder


def _keys_with_prefix(safetensors_path: str, prefix: str) -> list[str]:
    with safe_open(safetensors_path, "pt", "cpu") as f:
        return [k for k in f.keys() if k.startswith(prefix)]


def load_component(module: torch.nn.Module, safetensors_path: str, prefix: str,
                   *, allow_unexpected: bool = False) -> torch.nn.Module:
    """Load ``prefix``-namespaced tensors from a safetensors file into ``module``
    (prefix stripped). Raises on missing/unexpected keys (minus non-persistent
    rotary buffers)."""
    sd = {}
    with safe_open(safetensors_path, "pt", "cpu") as f:
        for k in f.keys():
            if k.startswith(prefix):
                sd[k[len(prefix):]] = f.get_tensor(k)
    if not sd:
        raise RuntimeError(f"load_component: no tensors under prefix {prefix!r} in {safetensors_path}")
    target_dtype = next(module.parameters()).dtype
    sd = {k: v.to(target_dtype) if v.is_floating_point() else v for k, v in sd.items()}
    missing, unexpected = module.load_state_dict(sd, strict=False)
    missing = [m for m in missing if not m.endswith("inv_freq")]
    if missing or (unexpected and not allow_unexpected):
        raise RuntimeError(
            f"load_component({prefix}): missing={missing[:6]} unexpected={unexpected[:6]}")
    return module


def dit_mode_from_ckpt(safetensors_path: str,
                       prefix: str = "velocity_field_predictor.") -> str:
    """meanflow iff the DiT carries a duration_embedder in the checkpoint."""
    has_duration = bool(_keys_with_prefix(safetensors_path, prefix + "duration_embedder."))
    return "meanflow" if has_duration else "flow_matching"


def build_native_dit(model_dir: str, *, device="cuda", dtype=torch.bfloat16) -> DiT:
    """Build + load the native DiT (velocity_field_predictor) from a model dir."""
    cfg = load_config(model_dir)
    tc = dit_config(model_dir)
    ckpt = os.path.join(model_dir, "model.safetensors")
    mode = dit_mode_from_ckpt(ckpt)
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(device):
            dit = DiT(in_dim=cfg["DiT"]["hidden_size"], out_dim=cfg["latent_dim"],
                      transformer_config=tc, mode=mode).eval()
    finally:
        torch.set_default_dtype(prev)
    load_component(dit, ckpt, "velocity_field_predictor.")
    return dit


def _llm_hidden_size(model_dir: str) -> int:
    import json
    with open(os.path.join(model_dir, "llm_config.json")) as f:
        return int(json.load(f)["hidden_size"])


def build_native_patch_encoder(model_dir: str, *, device="cuda",
                               dtype=torch.bfloat16) -> VAESemanticEncoder:
    """Build + load the native patch_encoder (VAESemanticEncoder) from a model dir."""
    cfg = load_config(model_dir)
    ckpt = os.path.join(model_dir, "model.safetensors")
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(device):
            pe = VAESemanticEncoder(
                in_dim=cfg["latent_dim"], out_dim=_llm_hidden_size(model_dir),
                config=PatchEncoderModelConfig(cfg)).eval()
    finally:
        torch.set_default_dtype(prev)
    load_component(pe, ckpt, "patch_encoder.")
    return pe
