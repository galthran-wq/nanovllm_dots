#!/usr/bin/env python
"""Streaming vocoder == one-shot vocoder, and produce a real wav.

The engine now vocodes incrementally (step_stream: one BigVGAN stream_step per
emitted patch + a flush at the end). This checks that the streamed audio
(concatenated chunks) reconstructs the SAME waveform as one-shot decoding the
full latent sequence (vocoder.inference_from_latents) -- the reference exposes
both paths and they must agree. Also writes the streamed wav so it can be heard.

Run:
    PYTHONPATH=stubs:. HF_HUB_OFFLINE=1 .venv/bin/python scripts/verify_stream_vocoder.py \
        --model models/dots.tts-mf --out /tmp/dots_stream.wav
"""
from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

EN1 = "Hello, this is a streaming synthesis test for the dots text to speech engine."


def set_seed(s: int) -> None:
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/dots.tts-mf")
    ap.add_argument("--num-steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="/tmp/dots_stream.wav")
    args = ap.parse_args()

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29528")
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

    set_seed(args.seed)
    eng = DotsBatchEngine(
        runtime, paged, model_dir=args.model, num_kvcache_blocks=256,
        block_size=256, max_num_seqs=8, fm_accel="cudagraph", flash_pe=True,
    )
    eng.add_request("r0", EN1, num_steps=args.num_steps, guidance_scale=1.2)

    chunks: list[torch.Tensor] = []
    n_audio = n_done = 0
    for sid, kind, data in eng.generate_stream():
        if kind == "audio":
            chunks.append(data); n_audio += 1
        elif kind == "done":
            n_done += 1

    streamed = torch.cat(chunks) if chunks else torch.zeros(0)
    latents = torch.cat(eng._results["r0"], dim=1)         # [1, frames, latent_dim]
    oneshot = eng.vocode(latents)

    sr = int(runtime.model.vocoder.sample_rate)
    n = min(streamed.numel(), oneshot.numel())
    cos = F.cosine_similarity(streamed[:n], oneshot[:n], dim=0).item() if n else 0.0
    dur = streamed.numel() / sr
    ok = cos > 0.999 and streamed.numel() > 0 and n_done == 1

    try:
        import soundfile as sf
        sf.write(args.out, streamed.numpy(), sr)
        wrote = args.out
    except Exception as e:                                  # pragma: no cover
        wrote = f"(save failed: {e})"

    print(f"[verify_stream_vocoder] patches/chunks={n_audio} done={n_done} "
          f"streamed={streamed.numel()} oneshot={oneshot.numel()} samples "
          f"dur={dur:.2f}s sr={sr} cos(stream,oneshot)={cos:.5f} "
          f"wav={wrote} {'PASS' if ok else 'FAIL'}")
    dist.destroy_process_group()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
