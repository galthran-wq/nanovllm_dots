# CLAUDE.md

Guidance for Claude Code working in this repo. Read alongside `README.md` (user-
facing overview + benchmarks).

## What this project is

A throughput-optimized inference engine + FastAPI server for **dots.tts**
(MeanFlow TTS). The runtime is a **native port** of every dots.tts neural net —
the server imports **zero `dots_tts`** at runtime. `reference/dots.tts` is a
dev/test-only dependency, kept solely so port tests can assert native ==
reference at cos ≈ 1.0. Do not reintroduce a runtime dependency on it.

## Layout

- `nanovllm_dots/models/dots/native/` — the native model: `dit.py`,
  `patch_encoder.py`, `vocoder/` (BigVGAN + alias-free ops), `speaker/` (CAM++),
  `core.py`, `model.py`, `runtime.py`, `pipeline/` (tokenizer + text frontend).
  Mirrors the reference structure **exactly** (same attribute names, **unfused
  q/k/v**) so the accelerators bind to the same leaves and weights load by direct
  `state_dict` copy. Don't fuse QKV here — it breaks the accelerators.
- `nanovllm_dots/models/dots/engine.py` — `DotsBatchEngine`, the continuous-
  batching driver. `fm_accel` selects the FM path:
  `none | compile | cudagraph | mfgraph | kvcache | hybrid`.
- `deployment/app/` — FastAPI server. `DotsStreamServer` lives in
  `deployment/app/server.py` (async worker-thread continuous batching + vocoder).
- `scripts/` — `bench_engine.py` (latent-only), `bench_server.py` (full async
  server), `bench_fm_crossover.py`, `gen_golden.py`.
- `tests/` — port tests + e2e (see Testing).
- `reference/dots.tts` — read/mirror only; never import at runtime.
- `stubs/tn` — shim replacing WeTextProcessing/pynini (don't add that dep back).

## Optimization map (where the perf lives)

Profiling: ~81 % of the work is the **FM/DiT head** — optimize there first.
- Production server path: `fm_accel=compile` + `flash_pe`. Compile (not
  cudagraph) because continuous batching changes batch shapes → graphs would
  re-capture, and whole-patch graphs OOM under concurrency.
- `mfgraph`/`cudagraph` — best **single-stream** RTF (0.57), but memory-heavy;
  OOMs at concurrency ≥ 8 on the shared GPU. Single-stream latency only.
- `kvcache` — O(P)/timestep; for **long** single streams (40 s+) where the graph
  OOMs. Loses on short audio (occupancy-bound, not FLOP-bound).
- `flash_pe` — batched patch encoder; raised the throughput ceiling ~4×.
- int8 DiT (`int8_dit.py`) is a **dead end** (~2× regression) — don't revisit.

## Testing

Tests are `advanced` (need GPU + weights under `models/`; auto-skip otherwise)
and split by dependency:
- `pytest -m "not oracle"` — default suite, runs on the **native** model, no
  `dots_tts` needed. Uses the `make_native_engine` fixture.
- `pytest -m oracle` — asserts native == reference (cos > 0.999). Any test using
  the `reference` fixture is auto-marked `oracle` (see
  `tests/conftest.py::pytest_collection_modifyitems`).
- `legacy` marker = superseded/soar-path tests.
- Port tests (`tests/test_native_*.py`) gate on patch-0 cos > 0.99 + liveness +
  length; RNG decorrelation after patch 0 is **expected**, not a bug.

## Environment & commands

- Run things with: `PYTHONPATH=stubs:. HF_HUB_OFFLINE=1 .venv/bin/python ...`
  (the `stubs:` prefix and offline flag are required for reference/scripts).
- Deps via **uv** (`uv sync`); `reference/dots.tts` installed `--no-deps`.
- Models live in `models/dots.tts-{mf,soar}`. `mf` (MeanFlow, NFE=4) is the
  shipping checkpoint; `soar` is flow-matching (legacy path).
- Latent → audio: 1 latent frame = 1920 samples @ 48 kHz = 0.04 s.

## Operational constraints (important)

- **Shared GPU.** One A100-40GB shared with other jobs. omnivoice PIDs (e.g.
  1322, 2880961) are **not ours — never kill them.** ~18 GiB is typically free;
  `mfgraph` at concurrency will OOM in that budget — expected.
- **Downloads bypass the proxy:** `env -u HTTP_PROXY -u HTTPS_PROXY ...` in a
  foreground bash (the box proxy is ~1.5 MiB/s vs ~46 MiB/s direct).
- **Git pushes:** the terminal's git-askpass socket (Cursor/VS Code) dies when
  the session disconnects → **push fails; the user pushes manually.** Commit
  locally; ask the user to push.
- **Author / commits:** author = `Lev Morozov <levmorozov900@gmail.com>`. End
  commit messages with `Co-Authored-By: Claude Opus 4.8 (1M context)
  <noreply@anthropic.com>`. Work in feature branches + PRs (the established loop:
  branch → subagent review → PR → user merges).
- Do **not** probe/extract the git credential helper for a token.
