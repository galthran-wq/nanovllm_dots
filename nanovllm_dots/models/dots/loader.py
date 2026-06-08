"""Weight loading for the dots.tts model parts that run through the engine.

The dots checkpoint (`model.safetensors`) stores the whole `DotsTtsCore` with
keys like `llm.model.layers.0.self_attn.q_proj.weight`. The engine LLM backbone
(`QwenLLM`) uses *fused* projections (qkv_proj, gate_up_proj), so we map the
separate q/k/v and gate/up tensors onto the fused params via the model's
`packed_modules_mapping` and the per-param `weight_loader` (same mechanism as
`nanovllm_dots.utils.loader.load_model`, but filtered to a key prefix).
"""
from __future__ import annotations

import torch
from safetensors import safe_open

from nanovllm_dots.utils.loader import default_weight_loader


def load_llm_weights(model, safetensors_path: str, prefix: str = "llm.model.") -> None:
    """Load `prefix`* tensors from a safetensors file into `model` (a QwenLLM).

    Keys are stripped of `prefix` (e.g. `llm.model.layers.0...` -> `layers.0...`)
    and routed through `packed_modules_mapping` for fused QKV / gate_up.
    """
    packed = getattr(model, "packed_modules_mapping", {})
    visited: set[str] = set()

    with safe_open(safetensors_path, "pt", "cpu") as f:
        for full_key in f.keys():
            if not full_key.startswith(prefix):
                continue
            name = full_key[len(prefix):]
            for src in packed:
                if src in name:
                    dst, shard_id = packed[src]
                    param_name = name.replace(src, dst)
                    param = model.get_parameter(param_name)
                    param.weight_loader(param, f.get_tensor(full_key), shard_id)
                    visited.add(param_name)
                    break
            else:
                param = model.get_parameter(name)
                loader = getattr(param, "weight_loader", default_weight_loader)
                loader(param, f.get_tensor(full_key))
                visited.add(name)

    missing = [n for n, _ in model.named_parameters() if n not in visited]
    if missing:
        raise ValueError(f"Missing LLM parameters after load: {missing[:8]}{'...' if len(missing) > 8 else ''}")
