"""FastAPI server for dots.tts streaming synthesis (Phase 3).

Wraps DotsStreamServer (one batched engine, continuous batching) behind HTTP:
  POST /generate         -> full WAV (audio/wav), one-shot vocode (high throughput)
  POST /generate_stream  -> chunked WAV streamed as the engine produces audio
  GET  /health           -> {"status": "ok"}

The DotsStreamServer is built once at startup (lifespan); the model dir comes from
the DOTS_MODEL env var (default models/dots.tts-mf). Run with scripts/serve.py.
"""
from __future__ import annotations

import os
import struct
from contextlib import asynccontextmanager
from typing import AsyncIterator

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from nanovllm_dots.models.dots.server import DotsStreamServer

_server: DotsStreamServer | None = None


class GenerateRequest(BaseModel):
    text: str
    num_steps: int = 4
    guidance_scale: float = 1.2
    eos_threshold: float = 0.8


def _pcm16(wav: torch.Tensor) -> bytes:
    a = wav.detach().cpu().numpy() if isinstance(wav, torch.Tensor) else np.asarray(wav)
    a = np.clip(a, -1.0, 1.0)
    return (a * 32767.0).astype("<i2").tobytes()


def _wav_header(sample_rate: int, data_bytes: int, *, channels: int = 1,
                bits: int = 16) -> bytes:
    """44-byte WAV header. For streaming, data_bytes/RIFF size are set to a max
    placeholder (0xFFFFFFFF) so the header can be emitted before the length is
    known -- players read until the stream ends."""
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    riff = 0xFFFFFFFF if data_bytes < 0 else 36 + data_bytes
    data = 0xFFFFFFFF if data_bytes < 0 else data_bytes
    return (b"RIFF" + struct.pack("<I", riff) + b"WAVE"
            + b"fmt " + struct.pack("<IHHIIHH", 16, 1, channels, sample_rate,
                                    byte_rate, block_align, bits)
            + b"data" + struct.pack("<I", data))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _server
    model_dir = os.environ.get("DOTS_MODEL", "models/dots.tts-mf")
    max_num_seqs = int(os.environ.get("DOTS_MAX_SEQS", "64"))
    _server = DotsStreamServer.from_pretrained(model_dir, max_num_seqs=max_num_seqs)
    yield
    _server.close()


app = FastAPI(title="dots.tts", lifespan=lifespan)


@app.get("/health")
async def health() -> JSONResponse:
    ok = _server is not None
    return JSONResponse({"status": "ok" if ok else "loading",
                         "sample_rate": _server.sample_rate if ok else None})


@app.post("/generate")
async def generate(req: GenerateRequest) -> Response:
    """Full waveform as a complete WAV (one-shot vocode)."""
    wav = await _server.generate_wav(
        req.text, num_steps=req.num_steps, guidance_scale=req.guidance_scale,
        eos_threshold=req.eos_threshold)
    pcm = _pcm16(wav)
    body = _wav_header(_server.sample_rate, len(pcm)) + pcm
    return Response(content=body, media_type="audio/wav")


@app.post("/generate_stream")
async def generate_stream(req: GenerateRequest) -> StreamingResponse:
    """Stream a WAV as audio is produced (header first, then PCM16 chunks)."""
    sr = _server.sample_rate

    async def body() -> AsyncIterator[bytes]:
        yield _wav_header(sr, -1)                       # placeholder-length header
        async for chunk in _server.generate(
                req.text, num_steps=req.num_steps,
                guidance_scale=req.guidance_scale,
                eos_threshold=req.eos_threshold, stream=True):
            yield _pcm16(chunk)

    return StreamingResponse(body(), media_type="audio/wav")
