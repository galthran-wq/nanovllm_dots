#!/usr/bin/env python
"""Phase 1, step 2: validate the paged-KV *incremental append* path.

`verify_llm.py` proved a single full-sequence prefill (no cache) matches HF.
This proves the thing the dots loop actually does: feed the sequence in
*chunks* through a growing paged KV cache (prefill chunk, then small per-patch
chunks) and recover hidden states identical to the one-shot forward. That is
the deferred decode/paged-cache validation (review item C2), isolated from the
rest of dots.

Run:
    PYTHONPATH=. .venv/bin/python scripts/verify_paged.py --model models/dots.tts-soar
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/dots.tts-soar")
    ap.add_argument(
        "--text",
        default="Hello, this is a reference sample generated for regression testing.",
    )
    ap.add_argument("--chunk", type=int, default=4, help="append chunk size after the prefill")
    args = ap.parse_args()

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29513")
    dist.init_process_group("nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)

    from transformers import AutoTokenizer, Qwen2Config

    from nanovllm_dots.models.dots.loader import load_llm_weights
    from nanovllm_dots.models.dots.model_llm import QwenLLM
    from nanovllm_dots.models.dots.paged_llm import PagedLLMRunner
    from nanovllm_dots.utils.context import reset_context, set_context

    model_dir = args.model
    ckpt = os.path.join(model_dir, "model.safetensors")
    cfg = Qwen2Config.from_json_file(os.path.join(model_dir, "llm_config.json"))

    tok = AutoTokenizer.from_pretrained(model_dir)
    ids = tok(args.text, return_tensors="pt").input_ids[0].cuda()
    seq_len = int(ids.shape[0])
    print(f"[verify_paged] seq_len={seq_len} chunk={args.chunk}")

    torch.set_default_dtype(torch.bfloat16)
    with torch.device("cuda"):
        llm = QwenLLM(cfg).eval()
    load_llm_weights(llm, ckpt)
    torch.set_default_dtype(torch.float32)

    with torch.no_grad():
        embeds = llm.embed_tokens(ids)  # [L, H]

    # --- reference: single full-sequence forward, no KV cache (slot=-1) ---
    positions = torch.arange(seq_len, device="cuda")
    cu = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")
    slot = torch.full((seq_len,), -1, dtype=torch.int32, device="cuda")
    set_context(True, cu, cu, seq_len, seq_len, slot, None, None)
    with torch.no_grad():
        full = llm(embeds, positions).float()
    reset_context()

    # --- incremental: prefill a prefix, then append `chunk`-sized blocks ---
    # Mirror dots: an initial multi-token prefill, then repeated small appends.
    runner = PagedLLMRunner(llm, block_size=256, max_seq_len=4096)
    prefill_len = max(1, seq_len // 2)
    pieces = [embeds[:prefill_len]]
    p = prefill_len
    while p < seq_len:
        pieces.append(embeds[p : p + args.chunk])
        p += args.chunk

    outs = []
    runner.reset()
    with torch.no_grad():
        for piece in pieces:
            outs.append(runner.append(piece).float())
    inc = torch.cat(outs, dim=0)  # [L, H]

    diff = (inc - full).abs()
    cos = torch.nn.functional.cosine_similarity(inc, full, dim=-1)
    print(f"[verify_paged] pieces={len(pieces)} (prefill={prefill_len}, +{args.chunk}-chunks)")
    print(f"[verify_paged] max|diff|={diff.max().item():.4e}  mean|diff|={diff.mean().item():.4e}")
    print(f"[verify_paged] cosine: min={cos.min().item():.6f}  mean={cos.mean().item():.6f}")
    ok = cos.min().item() > 0.999
    print(f"[verify_paged] {'PASS' if ok else 'FAIL'} (min cosine > 0.999)")

    dist.destroy_process_group()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
