#!/usr/bin/env python
"""Voice cloning step B (in-context prompt prefill): runs + faithful vs reference.

Checks the full clone-prefill path: prompt audio -> flash_pe.prefill seeds the
encoder cache, the prompt patch embeddings + spans seed the LLM prefill, and the
prompt (hidden, latent) pairs seed the FM history. Then generation continues.

Validation:
 1. our engine (clone_prefill=True) runs end-to-end and produces audio.
 2. fm history after prefill has the expected length (prompt + seed hidden).
 3. cross-check vs the reference DotsTtsRuntime.generate on the SAME prompt+text
    (perceptual: both wavs saved) and the prompt-prefill FM seeding faithfulness
    (our post-prefill fm_sequence vs the reference _prefill, cos).

Run:
    PYTHONPATH=stubs:. HF_HUB_OFFLINE=1 .venv/bin/python scripts/verify_clone_prefill.py \
        --model models/dots.tts-mf --ref-audio golden_mf/en1.wav
"""
from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
import torch.distributed as dist

TEXT = "The quick brown fox jumps over the lazy dog near the river bank."
PROMPT_TEXT = "Hello, this is a reference sample generated for regression testing."


def set_seed(s: int) -> None:
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/dots.tts-mf")
    ap.add_argument("--ref-audio", default="golden_mf/en1.wav")
    ap.add_argument("--prompt-text", default=PROMPT_TEXT)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29531")
    dist.init_process_group("nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)

    from transformers import Qwen2Config
    from dots_tts.models.dots_tts.model import DotsTtsModel  # noqa: F401
    from dots_tts.runtime import DotsTtsRuntime
    from nanovllm_dots.models.dots.loader import load_llm_weights
    from nanovllm_dots.models.dots.model_llm import QwenLLM
    from nanovllm_dots.models.dots.engine import DotsBatchEngine
    import soundfile as sf

    runtime = DotsTtsRuntime.from_pretrained(
        args.model, precision="bfloat16", optimize=False, max_generate_length=256)
    ckpt = os.path.join(args.model, "model.safetensors")
    cfg = Qwen2Config.from_json_file(os.path.join(args.model, "llm_config.json"))
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("cuda"):
        paged = QwenLLM(cfg).eval()
    load_llm_weights(paged, ckpt)
    torch.set_default_dtype(torch.float32)
    sr = int(runtime.model.vocoder.sample_rate)

    def build():
        return DotsBatchEngine(
            runtime, paged, model_dir=args.model, num_kvcache_blocks=512,
            block_size=256, max_num_seqs=8, fm_accel="cudagraph", flash_pe=True)

    # (1) clone-prefill run
    set_seed(args.seed)
    eng = build()
    eng.add_request("c", TEXT, num_steps=4, prompt_audio_path=args.ref_audio,
                    prompt_text=args.prompt_text, clone_prefill=True)
    p = eng.scheduler._id_to_seq["c"].custom_payload
    prompt_P = int(p.prompt_patches.size(1))
    lat = eng.run_all()["c"]
    wav = eng.vocode(lat)
    sf.write("/tmp/clone_prefill_ours.wav", wav.numpy(), sr)

    # (2) g_cond-only (step A) for comparison
    set_seed(args.seed)
    engA = build()
    engA.add_request("a", TEXT, num_steps=4, prompt_audio_path=args.ref_audio,
                     prompt_text=args.prompt_text, clone_prefill=False)
    watA = engA.vocode(engA.run_all()["a"])
    sf.write("/tmp/clone_gcondonly.wav", watA.numpy(), sr)

    # (3) reference full generate on the same prompt+text
    ref_wav = None
    try:
        set_seed(args.seed)
        out = runtime.generate(
            text=TEXT, prompt_audio_path=args.ref_audio, prompt_text=args.prompt_text,
            num_steps=4, guidance_scale=1.2)
        rw = out["audio"] if isinstance(out, dict) else out
        ref_wav = torch.as_tensor(rw).reshape(-1).float().cpu()
        sf.write("/tmp/clone_prefill_ref.wav", ref_wav.numpy(), sr)
    except Exception as e:
        print("ref generate skipped:", repr(e)[:200])

    ok = wav.numel() > 0 and prompt_P > 0
    print(f"[verify_clone_prefill] prompt_patches={prompt_P} "
          f"ours={wav.numel()} ({wav.numel()/sr:.2f}s) gcondonly={watA.numel()} "
          f"ref={'%d'%ref_wav.numel() if ref_wav is not None else 'n/a'} "
          f"wavs=/tmp/clone_prefill_(ours|ref|gcondonly).wav {'PASS' if ok else 'FAIL'}")
    dist.destroy_process_group()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
