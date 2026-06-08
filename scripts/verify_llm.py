#!/usr/bin/env python
"""Phase 1, step 1: validate the ported Qwen2 LLM backbone in isolation.

Runs a single-sequence prefill through the engine `QwenLLM` (paged Attention,
flash-attn varlen, no KV cache / no prefix cache) and compares the final hidden
states against a reference `transformers.Qwen2ForCausalLM` loaded from the same
checkpoint. High cosine similarity => the port (fused QKV/gate_up, RoPE theta,
qkv-bias, RMSNorm) is correct.

Run:
    .venv/bin/python scripts/verify_llm.py --model models/dots.tts-soar
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
from safetensors import safe_open


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/dots.tts-soar")
    ap.add_argument("--text", default="Hello, this is a reference sample generated for regression testing.")
    args = ap.parse_args()

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29512")
    dist.init_process_group("nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)

    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

    from nanovllm_dots.models.dots.loader import load_llm_weights
    from nanovllm_dots.models.dots.model_llm import QwenLLM
    from nanovllm_dots.utils.context import reset_context, set_context

    model_dir = args.model
    ckpt = os.path.join(model_dir, "model.safetensors")
    cfg = Qwen2Config.from_json_file(os.path.join(model_dir, "llm_config.json"))

    tok = AutoTokenizer.from_pretrained(model_dir)
    ids = tok(args.text, return_tensors="pt").input_ids[0].cuda()
    seq_len = int(ids.shape[0])
    positions = torch.arange(seq_len, device="cuda")
    print(f"[verify_llm] seq_len={seq_len}")

    # --- ported backbone ---
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("cuda"):
        llm = QwenLLM(cfg).eval()
    load_llm_weights(llm, ckpt)
    torch.set_default_dtype(torch.float32)

    cu = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")
    slot = torch.full((seq_len,), -1, dtype=torch.int32, device="cuda")  # no KV write
    set_context(True, cu, cu, seq_len, seq_len, slot, None, None)
    with torch.no_grad():
        embeds = llm.embed_tokens(ids)
        mine = llm(embeds, positions).float()  # [L, H]
    reset_context()

    # --- reference Qwen2 (loaded standalone from the same checkpoint) ---
    ref = Qwen2ForCausalLM(cfg)
    sd = {}
    with safe_open(ckpt, "pt", "cpu") as f:
        for k in f.keys():
            if k.startswith("llm."):
                sd[k[len("llm."):]] = f.get_tensor(k)
    missing, unexpected = ref.load_state_dict(sd, strict=False)
    missing = [m for m in missing if "lm_head" not in m]  # tied, unused
    if missing or unexpected:
        print(f"[verify_llm] WARN load_state_dict missing={missing[:4]} unexpected={unexpected[:4]}")
    ref = ref.to(torch.bfloat16).cuda().eval()
    with torch.no_grad():
        ref_h = ref.model(input_ids=ids.unsqueeze(0), use_cache=False).last_hidden_state[0].float()

    diff = (mine - ref_h).abs()
    cos = torch.nn.functional.cosine_similarity(mine, ref_h, dim=-1)
    print(f"[verify_llm] max|diff|={diff.max().item():.4e}  mean|diff|={diff.mean().item():.4e}")
    print(f"[verify_llm] ref abs mean={ref_h.abs().mean().item():.4e}")
    print(f"[verify_llm] cosine: min={cos.min().item():.6f}  mean={cos.mean().item():.6f}")
    ok = cos.min().item() > 0.99
    print(f"[verify_llm] {'PASS' if ok else 'FAIL'} (min cosine > 0.99)")

    dist.destroy_process_group()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
