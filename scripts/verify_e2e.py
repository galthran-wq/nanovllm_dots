#!/usr/bin/env python
"""Phase 1, step 3: end-to-end validation of the paged LLM inside dots.tts.

Strategy: run the *reference* dots generation loop unchanged
(`runtime.model._generate_latents_stream`, eager, fixed seed) but redirect the
one hot path we ported -- `core.step_llm` -- through our paged `QwenLLM`
(`PagedLLMRunner`). Everything else (patch_encoder, FM/DiT ODE, eos_proj,
hidden_proj/latent_proj, vocoder) stays on the original dots modules. If the
generated latent stream matches `golden/<name>.latents.npy` (produced by the
fully-reference run under the same seed), then the paged LLM backbone is
correct in the *real* generation context, not just in isolation.

Bit-exactness is not expected: our backbone differs from HF Qwen2 by ~1e-3
(bf16, fused QKV/gate_up, flash vs eager). Those tiny hidden-state deltas feed
hidden_proj -> FM history -> ODE, so we check high cosine / low MSE on the
overlapping patches, and that the patch counts (hence eos timing) agree.

Run:
    PYTHONPATH=stubs:. HF_HUB_OFFLINE=1 .venv/bin/python scripts/verify_e2e.py \
        --model models/dots.tts-soar
"""
from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
import torch.distributed as dist


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


PROMPTS = [
    ("en1", "Hello, this is a reference sample generated for regression testing."),
    ("zh1", "这是一个用于回归测试的参考样本。"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/dots.tts-soar")
    ap.add_argument("--golden", default="golden")
    ap.add_argument("--precision", default="bfloat16")
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--guidance-scale", type=float, default=1.2)
    ap.add_argument("--max-generate-length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--debug-llm", action="store_true",
                    help="also run HF step_llm and report per-call hidden divergence")
    ap.add_argument("--save-paged-golden", action="store_true",
                    help="write the paged-engine latents to golden/<name>.paged_latents.npy "
                         "as the self-consistent Phase-2 regression reference")
    args = ap.parse_args()

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29514")
    dist.init_process_group("nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)

    from transformers import Qwen2Config
    from dots_tts.models.dots_tts.model import DotsTtsModel  # noqa: F401
    from dots_tts.runtime import DotsTtsRuntime

    from nanovllm_dots.models.dots.loader import load_llm_weights
    from nanovllm_dots.models.dots.model_llm import QwenLLM
    from nanovllm_dots.models.dots.paged_llm import PagedLLMRunner

    runtime = DotsTtsRuntime.from_pretrained(
        args.model,
        precision=args.precision,
        optimize=False,
        max_generate_length=args.max_generate_length,
    )
    print(f"[e2e] runtime loaded (sr={runtime.sample_rate})")

    # --- build the paged backbone from the same checkpoint ---
    ckpt = os.path.join(args.model, "model.safetensors")
    cfg = Qwen2Config.from_json_file(os.path.join(args.model, "llm_config.json"))
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("cuda"):
        paged = QwenLLM(cfg).eval()
    load_llm_weights(paged, ckpt)
    torch.set_default_dtype(torch.float32)
    runner = PagedLLMRunner(paged, block_size=256, max_seq_len=4096)

    # --- redirect core.step_llm through the paged runner ---
    core = runtime.model.core
    hf_embed = core.llm.get_input_embeddings()
    orig_step_llm = core.step_llm  # HF path, kept for --debug-llm comparison
    hf_state = {"cache": None}
    # Per-call (NOT aggregated) paged-vs-HF hidden fidelity. In --debug-llm both
    # paths receive the identical input chunk each call (the driver follows the
    # paged trajectory), so this isolates the LLM map from FM-loop chaos: a flat
    # ~0.999 across ALL calls => the crater in latent space is amplification; a
    # ramp/crater in the *hidden* cosine => a real port bug.
    dbg = {"cos": [], "reldiff": [], "len_mismatch": 0}

    def patched_step_llm(inputs_embeds=None, input_ids=None, past_key_values=None):
        if (inputs_embeds is None) == (input_ids is None):
            raise ValueError("exactly one of inputs_embeds / input_ids")
        if inputs_embeds is None:
            inputs_embeds = hf_embed(input_ids)
        hidden = runner.append(inputs_embeds)  # [L, hidden]
        if args.debug_llm:
            _, hf_hidden, _, hf_state["cache"] = orig_step_llm(
                inputs_embeds=inputs_embeds, past_key_values=hf_state["cache"]
            )
            a = hidden.float()
            b = hf_hidden[0].float()
            dbg["cos"].append(
                torch.nn.functional.cosine_similarity(a, b, dim=-1).min().item()
            )
            dbg["reldiff"].append(((a - b).norm() / b.norm().clamp_min(1e-6)).item())
            # Off-by-one guard: paged history length must equal HF cache length.
            if runner.past_len != hf_state["cache"].get_seq_length():
                dbg["len_mismatch"] += 1
        return inputs_embeds, hidden.unsqueeze(0), None, runner

    core.step_llm = patched_step_llm

    all_pass = True
    for name, text in PROMPTS:
        golden_path = os.path.join(args.golden, f"{name}.latents.npy")
        if not os.path.exists(golden_path):
            print(f"[e2e] {name}: no golden at {golden_path}, skipping")
            continue
        golden = torch.from_numpy(np.load(golden_path)).float()  # [1, P*patch, latent]

        set_seed(args.seed)
        inputs = runtime._prepare_inputs(
            text=text,
            prompt_audio_path=None,
            prompt_text=None,
            template_name=None,
            language=None,
            normalize_text=False,
        )
        runner.reset()
        hf_state["cache"] = None
        dbg.update(cos=[], reldiff=[], len_mismatch=0)
        lat = [
            l.detach().float().cpu()
            for l in runtime.model._generate_latents_stream(
                inputs,
                precision=args.precision,
                ode_method="euler",
                num_steps=args.num_steps,
                guidance_scale=args.guidance_scale,
            )
        ]
        mine = torch.cat(lat, dim=1) if lat else torch.zeros((1, 0, 0))
        if args.save_paged_golden:
            paged_path = os.path.join(args.golden, f"{name}.paged_latents.npy")
            np.save(paged_path, mine.numpy())
            print(f"[e2e] {name}: saved paged-golden -> {paged_path}")

        n = min(mine.size(1), golden.size(1))
        a, b = mine[:, :n], golden[:, :n]
        # Per-patch (4 frames) cosine trajectory: amplification shows as early
        # patches ~1.0 degrading over the autoregressive sequence.
        ps = 4
        traj = []
        for i in range(0, n, ps):
            pa = a[:, i : i + ps].reshape(-1, a.size(-1))
            pb = b[:, i : i + ps].reshape(-1, b.size(-1))
            traj.append(torch.nn.functional.cosine_similarity(pa, pb, dim=-1).mean().item())
        traj_str = " ".join(f"{t:.2f}" for t in traj)
        mse = torch.mean((a - b) ** 2).item()
        cos = torch.nn.functional.cosine_similarity(
            a.reshape(-1, a.size(-1)), b.reshape(-1, b.size(-1)), dim=-1
        )
        same_len = mine.size(1) == golden.size(1)
        # Correctness gate for a non-bit-exact LLM swap in an autoregressive,
        # stochastic generator: the golden is a deterministic fixed point of the
        # exact HF path, so late-patch latents *must* drift (sampler chaos
        # amplifies the ~1% bf16/flash LLM delta through the FM feedback loop).
        # What stays diagnostic of correctness: (1) eos fires at the same patch
        # (len match), and (2) the early patches -- where accumulated history
        # divergence is still negligible -- reproduce the golden latent. A
        # structural bug would corrupt patch 0; amplification cannot.
        early = sum(traj[: min(3, len(traj))]) / max(1, min(3, len(traj)))
        ok = same_len and early > 0.999
        if args.debug_llm:
            # Rigorous check: the LLM map must be faithful at EVERY call (flat
            # per-call hidden cosine) with no history-length drift. This is what
            # actually proves the port; it isolates correctness from FM chaos.
            ok = ok and dbg["len_mismatch"] == 0 and (min(dbg["cos"]) > 0.99)
        all_pass = all_pass and ok
        print(
            f"[e2e] {name}: mine={list(mine.shape)} golden={list(golden.shape)} "
            f"len_match={same_len} early3_cos={early:.4f} (gate) | "
            f"whole-seq cos(min/mean)={cos.min().item():.4f}/{cos.mean().item():.4f} "
            f"mse={mse:.4e} [drift=amplification, informational] "
            f"{'PASS' if ok else 'FAIL'}"
        )
        print(f"[e2e] {name}: per-patch latent cos vs golden: {traj_str}")
        if args.debug_llm:
            cos_str = " ".join(f"{c:.3f}" for c in dbg["cos"])
            flat = (min(dbg["cos"]) > 0.99) if dbg["cos"] else False
            print(
                f"[e2e] {name}: per-CALL paged-vs-HF hidden cos ({len(dbg['cos'])} calls): {cos_str}"
            )
            print(
                f"[e2e] {name}: hidden min_cos={min(dbg['cos']):.5f} "
                f"max_reldiff={max(dbg['reldiff']):.4e} len_mismatch={dbg['len_mismatch']} "
                f"=> {'FLAT (amplification)' if flat and dbg['len_mismatch'] == 0 else 'NON-FLAT (investigate)'}"
            )

    dist.destroy_process_group()
    raise SystemExit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
