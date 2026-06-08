#!/usr/bin/env python
"""Phase 0: generate a golden reference (wav + latents + profile) from the
reference dots.tts implementation in eager mode, with fixed seeds, so the
optimized engine can be regression-tested against it.

Usage:
    .venv/bin/python scripts/gen_golden.py \
        --model rednote-hilab/dots.tts-soar \
        --out golden/ \
        [--prompt-audio ref.wav --prompt-text "..."]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# A couple of fixed prompts covering EN + ZH (dots uses raw BPE, no phonemes).
PROMPTS = [
    ("en1", "Hello, this is a reference sample generated for regression testing."),
    ("zh1", "这是一个用于回归测试的参考样本。"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="rednote-hilab/dots.tts-soar")
    ap.add_argument("--out", default="golden")
    ap.add_argument("--precision", default="bfloat16")
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--guidance-scale", type=float, default=1.2)
    ap.add_argument("--max-generate-length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--prompt-audio", default=None)
    ap.add_argument("--prompt-text", default=None)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import soundfile as sf
    from dots_tts.models.dots_tts.model import DotsTtsModel  # noqa: F401
    from dots_tts.runtime import DotsTtsRuntime

    t0 = time.time()
    runtime = DotsTtsRuntime.from_pretrained(
        args.model,
        precision=args.precision,
        optimize=False,  # eager: clean reference, no torch.compile / cudagraph
        max_generate_length=args.max_generate_length,
    )
    print(f"[golden] runtime loaded in {time.time() - t0:.1f}s, sr={runtime.sample_rate}")

    manifest = {
        "model": args.model,
        "precision": args.precision,
        "num_steps": args.num_steps,
        "guidance_scale": args.guidance_scale,
        "max_generate_length": args.max_generate_length,
        "seed": args.seed,
        "sample_rate": runtime.sample_rate,
        "items": [],
    }

    for name, text in PROMPTS:
        # Capture the raw latent sequence (pre-vocoder) for tight numeric checks,
        # re-running the latent stream under the same seed used for audio.
        set_seed(args.seed)
        inputs = runtime._prepare_inputs(
            text=text,
            prompt_audio_path=args.prompt_audio,
            prompt_text=args.prompt_text,
            template_name=None,
            language=None,
            normalize_text=False,
        )
        latents = [
            lat.detach().float().cpu()
            for lat in runtime.model._generate_latents_stream(
                inputs,
                precision=args.precision,
                ode_method="euler",
                num_steps=args.num_steps,
                guidance_scale=args.guidance_scale,
            )
        ]
        latent_arr = torch.cat(latents, dim=1).numpy() if latents else np.zeros((1, 0, 0))

        # Full audio under the same seed (separate run; vocoder is deterministic).
        set_seed(args.seed)
        res = runtime.generate(
            text=text,
            prompt_audio_path=args.prompt_audio,
            prompt_text=args.prompt_text,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            profile_inference=True,
        )
        audio = res["audio"].squeeze().detach().float().cpu().numpy()

        wav_path = out / f"{name}.wav"
        lat_path = out / f"{name}.latents.npy"
        sf.write(wav_path, audio, runtime.sample_rate)
        np.save(lat_path, latent_arr)
        item = {
            "name": name,
            "text": text,
            "wav": wav_path.name,
            "latents": lat_path.name,
            "latent_shape": list(latent_arr.shape),
            "audio_samples": int(audio.shape[-1]),
            "rtf": res["rtf"],
            "time_used": res["time_used"],
        }
        manifest["items"].append(item)
        print(f"[golden] {name}: latents={latent_arr.shape} samples={audio.shape[-1]} rtf={res['rtf']:.3f}")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"[golden] wrote {out/'manifest.json'}")


if __name__ == "__main__":
    main()
