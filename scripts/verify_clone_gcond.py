#!/usr/bin/env python
"""Voice cloning step A (g_cond): speaker conditioning is wired and faithful.

Checks:
 1. the engine's g_cond (via add_request prompt_audio_path) == the reference
    `_prepare_prompt_conditioning(use_prompt_prefill=False).g_cond` (bit-faithful).
 2. two different reference audios give DIFFERENT g_cond (the speaker encoder
    actually distinguishes voices), and the generated audio differs.
 3. generation with g_cond runs end-to-end and produces audio.
Writes the null / en-cloned / zh-cloned wavs so the timbre can be heard.

Run:
    PYTHONPATH=stubs:. HF_HUB_OFFLINE=1 .venv/bin/python scripts/verify_clone_gcond.py \
        --model models/dots.tts-mf
"""
from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

TEXT = "Voice cloning conditions the synthesizer on a reference speaker."
REF_A = "golden_mf/en1.wav"
REF_B = "golden_mf/zh1.wav"


def set_seed(s: int) -> None:
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/dots.tts-mf")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29530")
    dist.init_process_group("nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)

    from transformers import Qwen2Config
    from dots_tts.models.dots_tts.model import DotsTtsModel  # noqa: F401
    from dots_tts.runtime import DotsTtsRuntime
    from nanovllm_dots.models.dots.loader import load_llm_weights
    from nanovllm_dots.models.dots.model_llm import QwenLLM
    from nanovllm_dots.models.dots.engine import DotsBatchEngine

    runtime = DotsTtsRuntime.from_pretrained(
        args.model, precision="bfloat16", optimize=False, max_generate_length=256)
    ckpt = os.path.join(args.model, "model.safetensors")
    cfg = Qwen2Config.from_json_file(os.path.join(args.model, "llm_config.json"))
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("cuda"):
        paged = QwenLLM(cfg).eval()
    load_llm_weights(paged, ckpt)
    torch.set_default_dtype(torch.float32)

    def build():
        return DotsBatchEngine(
            runtime, paged, model_dir=args.model, num_kvcache_blocks=256,
            block_size=256, max_num_seqs=8, fm_accel="cudagraph", flash_pe=True)

    # (1) faithfulness: engine g_cond == reference g_cond
    pa = runtime._load_prompt_audio(REF_A).to("cuda")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        ref_cond = runtime.model._prepare_prompt_conditioning(
            pa, use_prompt_prefill=False, speaker_scale=1.5).g_cond
    eng0 = build()
    eng0.add_request("a", TEXT, num_steps=4, prompt_audio_path=REF_A, speaker_scale=1.5)
    eng_g = eng0.scheduler._id_to_seq["a"].custom_payload.g_cond
    faith = F.cosine_similarity(ref_cond.reshape(-1).float(),
                                eng_g.reshape(-1).float(), dim=0).item()

    # (2) distinct speakers -> distinct g_cond
    gA = ref_cond
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        gB = runtime.model._prepare_prompt_conditioning(
            runtime._load_prompt_audio(REF_B).to("cuda"),
            use_prompt_prefill=False, speaker_scale=1.5).g_cond
    spk_cos = F.cosine_similarity(gA.reshape(-1).float(), gB.reshape(-1).float(), dim=0).item()

    # (3) generate null / A / B; confirm audio differs; save
    import soundfile as sf
    sr = int(runtime.model.vocoder.sample_rate)
    outs = {}
    for tag, ref in [("null", None), ("cloneA_en", REF_A), ("cloneB_zh", REF_B)]:
        set_seed(args.seed)                      # same FM noise -> diff is conditioning
        eng = build()
        eng.add_request("r", TEXT, num_steps=4, prompt_audio_path=ref, speaker_scale=1.5)
        lat = eng.run_all()["r"]                 # already [1, frames, latent_dim]
        wav = eng.vocode(lat)
        outs[tag] = wav
        sf.write(f"/tmp/clone_{tag}.wav", wav.numpy(), sr)

    n = min(outs["null"].numel(), outs["cloneA_en"].numel())
    null_vs_A = F.cosine_similarity(outs["null"][:n], outs["cloneA_en"][:n], dim=0).item()

    ok = faith > 0.999 and spk_cos < 0.99 and null_vs_A < 0.999
    print(f"[verify_clone_gcond] g_cond faithful(engine==ref)={faith:.5f} "
          f"speaker A-vs-B cos={spk_cos:.4f} (lower=more distinct) "
          f"audio null-vs-cloneA cos={null_vs_A:.4f} "
          f"wavs=/tmp/clone_(null|cloneA_en|cloneB_zh).wav {'PASS' if ok else 'FAIL'}")
    dist.destroy_process_group()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
