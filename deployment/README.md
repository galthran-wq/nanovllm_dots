# dots.tts deployment

FastAPI HTTP server for the `nanovllm_dots` dots.tts engine. The async serving
wrapper (`DotsStreamServer` — continuous-batching engine + worker thread) lives
in `deployment/app/server.py`; the rest of this package is the thin HTTP layer
on top, mirroring the `nano-vllm-voxcpm` `deployment/` layout. `nanovllm_dots/`
is the GPU engine library it drives.

## Layout

```
deployment/
  app/
    main.py            create_app() + module-level `app`
    core/config.py     env-var config (ServiceConfig)
    core/lifespan.py   builds DotsStreamServer.from_pretrained -> app.state.server
    api/api.py         router composition
    api/deps.py        get_server (503 until ready)
    api/routes/        health.py, info.py, generate.py
    schemas/tts.py     GenerateRequest + responses
    services/wav.py    PCM16 / WAV-header helpers
  Dockerfile
```

Dependencies are managed with **uv** (`pyproject.toml` + `uv.lock` at the repo
root).

## API

| Method | Path               | Body            | Response                         |
|--------|--------------------|-----------------|----------------------------------|
| GET    | `/health`          | –               | `{"status":"ok"}` (liveness)     |
| GET    | `/ready`           | –               | 200 once loaded, else 503        |
| GET    | `/info`            | –               | sample_rate + service defaults   |
| POST   | `/generate`        | `GenerateRequest` | full `audio/wav` (one-shot)    |
| POST   | `/generate_stream` | `GenerateRequest` | chunked `audio/wav` (low TTFB) |

`GenerateRequest`: `text` (required), `num_steps=4`, `guidance_scale=1.2`,
`eos_threshold=0.8`. Voice cloning (optional): pass the reference audio as
`prompt_audio_base64` (+ `prompt_audio_format`, the HTTP-native way), plus
`prompt_text` (reference transcript), `speaker_scale=1.5`, `clone_prefill=true`.

`prompt_audio_path` (a path on the *server* filesystem) is also accepted but
**disabled by default** — set `DOTS_ALLOW_SERVER_AUDIO_PATH=1` to enable it
(trusted callers only; it lets the request name an arbitrary server-side file).

## Run locally

```bash
PYTHONPATH=stubs:. HF_HUB_OFFLINE=1 .venv/bin/python scripts/serve.py \
    --model models/dots.tts-mf --host 0.0.0.0 --port 8000
```

```bash
curl -s -X POST localhost:8000/generate \
     -H 'content-type: application/json' \
     -d '{"text":"hello world"}' -o out.wav

# voice clone over HTTP (base64 reference):
B64=$(base64 -w0 golden_mf/en1.wav)
curl -s -X POST localhost:8000/generate \
     -H 'content-type: application/json' \
     -d "{\"text\":\"hello world\",\"prompt_audio_base64\":\"$B64\",\"prompt_text\":\"...\"}" \
     -o clone.wav
```

## Docker

The model is reimplemented natively (`nanovllm_dots/models/dots/native`), so the
image has **no runtime dependency on the reference `dots_tts` package** — only
`stubs/` (the `tn` stand-in) must be present in the build context. Weights are
mounted, not baked.

```bash
docker build -f deployment/Dockerfile -t dots-tts .
docker run --gpus all -p 8000:8000 -v "$PWD/models:/models" dots-tts
```

Config via env (see `app/core/config.py`): `DOTS_MODEL` (default
`/models/dots.tts-mf` in the image), `DOTS_MAX_SEQS`, `DOTS_FM_ACCEL`
(`compile`|`cudagraph`), `DOTS_FLASH_PE`, `DOTS_GRAPH_DECODE`, `DOTS_WARMUP`,
`DOTS_HOST`, `DOTS_PORT`.

Needs the NVIDIA Container Toolkit on the host (for `--gpus`).

### Notes
- **No CUDA base image (~13 GB total).** The base is `python:3.12-slim`: the
  `torch +cu126` wheels bundle their own CUDA runtime (cudnn/cublas/…), flash-attn
  is a prebuilt wheel, and Triton ships its own `ptxas`, so the only host
  dependency is the NVIDIA *driver* (injected by `--gpus`). A CUDA base would ship
  a second, redundant copy of CUDA (it made the image ~36 GB).
- Deps via `uv sync --frozen --no-cache` from `uv.lock` (`--no-cache` keeps uv's
  ~7 GB wheel cache out of the image). The reference `dots_tts` package is NOT
  installed — the model is native; the input pipeline's `tn` text-normalizer is
  the lightweight `stubs/tn` stand-in (on PYTHONPATH).
- The prebuilt flash-attn wheel is torch2.8 / cu12 / cp312 / sm_80. To target a
  different GPU/torch, swap the wheel URL in `pyproject.toml`
  (`[tool.uv.sources]`) and re-run `uv lock`.
