#!/usr/bin/env python
"""End-to-end: the flash_pe ENGINE path == reference decode_patch, per patch.

verify_flash_pe.py validates the kernel on synthetic patches. This validates the
ENGINE INTEGRATION -- the row allocation/recycling, the denormalize(latents) batch,
and the history plumbing -- by driving the real engine (flash_pe=True) and, inside
its batched patch_encoder call, ALSO running the reference per-seq decode_patch on
a shadow decode-state kept in lock-step, then comparing the embeds row by row on
the LIVE trajectory. Multiple concurrent requests of different lengths exercise the
per-row cache_seqlens / cache_batch_idx path.

Run:
    PYTHONPATH=stubs:. HF_HUB_OFFLINE=1 .venv/bin/python scripts/verify_flash_pe_engine.py \
        --model models/dots.tts-mf --num-reqs 3
"""
from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

EN1 = "Hello, this is a reference sample generated for regression testing."
ZH1 = "这是一个用于回归测试的参考样本。"


def set_seed(s: int) -> None:
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/dots.tts-mf")
    ap.add_argument("--num-reqs", type=int, default=3)
    ap.add_argument("--num-steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29527")
    dist.init_process_group("nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)

    from transformers import Qwen2Config
    from dots_tts.models.dots_tts.model import DotsTtsModel  # noqa: F401
    from dots_tts.runtime import DotsTtsRuntime
    from nanovllm_dots.models.dots.loader import load_llm_weights
    from nanovllm_dots.models.dots.model_llm import QwenLLM
    from nanovllm_dots.models.dots.engine import DotsBatchEngine

    runtime = DotsTtsRuntime.from_pretrained(
        args.model, precision="bfloat16", optimize=False, max_generate_length=256
    )
    ckpt = os.path.join(args.model, "model.safetensors")
    cfg = Qwen2Config.from_json_file(os.path.join(args.model, "llm_config.json"))
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("cuda"):
        paged = QwenLLM(cfg).eval()
    load_llm_weights(paged, ckpt)
    torch.set_default_dtype(torch.float32)

    pe = runtime.model.core.patch_encoder
    dev = torch.device("cuda")
    dt = torch.bfloat16

    class DualEngine(DotsBatchEngine):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._shadow: dict[int, object] = {}   # pe_row -> reference decode-state
            self.coss: list[float] = []

        def _finish_patches_flash(self, seqs, latents, stops):
            core, dots = self.core, self.dots
            # reproduce the engine's row claim + flash embed
            rows = []
            for seq, latent in zip(seqs, latents):
                st = seq.custom_payload
                dots._append_history_chunk(st.gen_state, latent.unsqueeze(0))
                if st.pe_row is None:
                    st.pe_row = self._pe_free.pop()
                    self._flash_pe.reset_rows(
                        torch.tensor([st.pe_row], device=self.device, dtype=torch.int32))
                    self._shadow[st.pe_row] = pe.init_decode_state(
                        max_audio_patch_count=256, batch_size=1, device=dev, dtype=dt)
                rows.append(st.pe_row)
            rows_t = torch.tensor(rows, device=self.device, dtype=torch.int32)
            patches = core.io_helper.denormalize(latents)
            embeds = self._flash_pe.decode_patch(patches, rows_t)
            # reference per-seq on the SAME denormalized patches
            for i, (seq, latent, stop) in enumerate(zip(seqs, latents, stops)):
                st = seq.custom_payload
                sh = self._shadow[st.pe_row]
                positions = torch.arange(pe.out_ds_rate, device=dev, dtype=torch.long) + sh.seq_len
                ref_emb, conv_tail = pe.decode_patch(
                    patches[i:i + 1], sh.conv_tail, sh.layer_caches, positions)
                sh.conv_tail.copy_(conv_tail)
                sh.seq_len += pe.out_ds_rate
                c = F.cosine_similarity(ref_emb.reshape(-1).float(),
                                        embeds[i].reshape(-1).float(), dim=-1)
                self.coss.append(c.item())
                self._emit_patch(seq, st, latent.unsqueeze(0), embeds[i], stop)

    set_seed(args.seed)
    eng = DualEngine(
        runtime, paged, model_dir=args.model, num_kvcache_blocks=256,
        block_size=256, max_num_seqs=8, fm_accel="cudagraph", flash_pe=True,
    )
    texts = ([EN1, ZH1] * ((args.num_reqs + 1) // 2))[:args.num_reqs]
    for i, t in enumerate(texts):
        eng.add_request(f"r{i}", t, num_steps=args.num_steps, guidance_scale=1.2)
    out = eng.run_all()

    coss = eng.coss
    cmin, cmean = min(coss), sum(coss) / len(coss)
    lens = {k: v.size(1) for k, v in out.items()}
    ok = cmin > 0.999 and all(v > 0 for v in lens.values())
    print(f"[verify_flash_pe_engine] reqs={args.num_reqs} embeds={len(coss)} "
          f"flash-engine vs reference cos(min/mean)={cmin:.5f}/{cmean:.5f} "
          f"lens={lens} {'PASS' if ok else 'FAIL'}")
    dist.destroy_process_group()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
