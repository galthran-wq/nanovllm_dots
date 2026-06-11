# nanovllm-dots

A throughput-optimized inference engine and FastAPI server for **dots.tts**
(MeanFlow TTS). Built in the spirit of nano-vLLM: paged-KV attention, continuous
batching, and a stack of FM/DiT accelerators that take single-stream synthesis
comfortably below real time and push aggregate throughput past **40× real time**
on one A100.

The runtime is a **native re-implementation** of every dots.tts neural net — the
server imports zero `dots_tts` code. `reference/dots.tts` is kept only as a
dev/test dependency so the port can be validated bit-for-bit (cos ≈ 1.0).

## What's inside

```
nanovllm_dots/                 the engine library
  models/dots/
    native/                    native model port (DiT, patch_encoder, BigVGAN
                               vocoder, CAM++ speaker, Qwen2 backbone, pipeline)
    engine.py                  DotsBatchEngine — continuous-batching driver
    paged_llm.py batched_llm.py   paged-KV Qwen2 backbone
    batched_fm.py cached_fm.py flash_cached_fm.py
    cudagraph_dit.py graphed_meanflow.py   FM/DiT accelerators
    flash_patch_encoder.py     batched VAE patch encoder
deployment/                    FastAPI HTTP server (DotsStreamServer)
scripts/                       benchmarks (bench_engine, bench_server, ...)
tests/                         native-vs-reference port tests + e2e
reference/                     dots.tts (dev/test only — never imported at runtime)
```

## Pipeline & where the time goes

Each request runs: **Qwen2-1.5B backbone** (text → audio-token hidden states) →
**FM/DiT head** (flow-matching / MeanFlow solver, latent patches) → **BigVGAN
vocoder** (latents → 48 kHz audio). Profiling puts ~**81 %** of the work in the
FM head, which is where the optimizations are concentrated.

## Optimizations

| Stage | Optimization | Files | Effect |
|---|---|---|---|
| **FM/DiT** | Batched FM across requests | `batched_fm.py` | the core throughput lever — batches the 81 % hot path |
| FM/DiT | `torch.compile` of the DiT | `engine.py` (`fm_accel=compile`) | ~1.2× single-stream + scales cleanly under batching |
| FM/DiT | Whole-patch CUDA graph (full NFE solver captured) | `cudagraph_dit.py`, `graphed_meanflow.py` (`mfgraph`) | best single-stream RTF (**0.57**), exact vs eager; memory-heavy → not for high concurrency |
| FM/DiT | O(P²)→O(P) FM KV-cache | `cached_fm.py`, `flash_cached_fm.py` (`kvcache`) | holds RTF on **long** audio (40 s+) where the graph OOMs |
| FM/DiT | MeanFlow (NFE=4, CFG-distilled) | `native/core.py` | single-stream floor RTF **0.25** |
| **Patch enc.** | Flash/varlen **batched** patch encoder | `flash_patch_encoder.py` (`flash_pe`) | raised the throughput ceiling ~4× (15→57 streams @ RTF 1) |
| **LLM** | Paged-KV + continuous batching | `paged_llm.py`, `batched_llm.py` | ragged sequences in one varlen call |
| LLM | `flash_attn_with_kvcache` decode + CUDA-graph | `batched_llm.py` (`graph_decode`) | cheaper 1-token decode steps |
| **Vocoder** | Streaming BigVGAN (per-patch + flush) | `native/vocoder/` | low TTFB; per-patch vocode trades ~11× compute for latency |
| — | ~~int8 DiT~~ | `int8_dit.py` | **dead end** — ~2× regression, not used |

> The production server uses `fm_accel=compile` (not `cudagraph`): under
> continuous batching the graph would re-capture on every changing batch shape,
> and whole-patch graphs OOM under concurrency. `cudagraph`/`mfgraph` are for
> single-stream latency; `kvcache` is for very long single streams.

## Benchmarks

Measured 2026-06-11 on one **A100-PCIE-40GB** (shared — ~18 GiB free), `dots.tts-mf`,
NFE=4, torch 2.8/cu126. Reproduce with the commands below.

**Single-stream latency** (RTF = wall / audio seconds; lower is faster):

| FM path | RTF | note |
|---|---|---|
| eager (`none`) | 0.812 | baseline |
| `compile` | 0.675 | server default |
| `mfgraph` (whole-patch graph) | 0.569 | best latency |

**Latent-only engine throughput** (×RT, vocoder excluded) — eager vs the server path:

| concurrency | eager `none` | `compile` + `flash_pe` | speedup |
|---|---|---|---|
| 1 | 1.23 | 1.48 | 1.2× |
| 8 | 4.56 | 8.25 | 1.8× |
| 32 | 7.34 | 23.9 | 3.3× |
| 64 | 8.11 | **40.4** | **5.0×** |

**Full server** (`DotsStreamServer`, vocoder included, async continuous batching):

| concurrency | one-shot ×RT | RPS | RTF median | streaming ×RT | TTFB median |
|---|---|---|---|---|---|
| 1 | 3.9 | 1.1 | 0.256 | 1.8 | 0.19 s |
| 8 | 12.8 | 3.7 | 0.554 | 2.9 | 0.43 s |
| 32 | 18.7 | 5.3 | 1.323 | 3.2 | 1.53 s |
| 64 | **21.7** | 6.0 | 2.017 | — | — |

One-shot maximizes throughput; streaming minimizes time-to-first-byte (per-patch
vocode is ~11× costlier, so its aggregate throughput is lower by design.)

## Quickstart

Dependencies are managed with **uv** (`pyproject.toml` + `uv.lock`):

```sh
uv sync
# reference/dots.tts is installed --no-deps for tests only (see pyproject)
```

Run the server:

```sh
PYTHONPATH=. uv run uvicorn deployment.app.main:app --host 0.0.0.0 --port 8000
# or the container:  docker build -f deployment/Dockerfile -t dots-tts .
```

Generate (one-shot WAV / low-latency stream):

```sh
curl -s localhost:8000/generate        -d '{"text":"Hello there."}' -o out.wav
curl -s localhost:8000/generate_stream -d '{"text":"Hello there."}' -o out.wav
```

See `deployment/README.md` for the full API (voice cloning via
`prompt_audio_base64`, `/info`, `/ready`).

## Benchmarks & tests

```sh
# fresh numbers (the tables above)
PYTHONPATH=stubs:. HF_HUB_OFFLINE=1 .venv/bin/python scripts/bench_server.py \
    --model models/dots.tts-mf --concurrency 1,8,32,64 --num-steps 4
PYTHONPATH=stubs:. HF_HUB_OFFLINE=1 .venv/bin/python scripts/bench_engine.py \
    --model models/dots.tts-mf --num-steps 4 --fm-accel compile --flash-pe \
    --concurrency 1,8,32,64

# tests — default suite runs on the NATIVE model, no dots_tts needed:
pytest -m "not oracle"
# the oracle suite asserts native == reference (needs reference/dots.tts):
pytest -m oracle
```

Tests are marked `advanced` (need GPU + weights, auto-skip otherwise) and split
into **native** (default) vs **oracle** (uses the `reference` fixture). The port
tests proving native == reference at cos ≈ 1.0 are the oracle set; everything
else runs against the native package alone.
