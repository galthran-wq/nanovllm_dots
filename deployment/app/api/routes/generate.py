from __future__ import annotations

import base64
import binascii
import contextlib
import os
import tempfile
from typing import Any, AsyncIterator, Callable

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from starlette.background import BackgroundTask

from deployment.app.api.deps import get_server
from deployment.app.schemas.tts import GenerateRequest
from deployment.app.services.wav import pcm16, wav_header

router = APIRouter(tags=["generation"])


def _resolve_prompt_audio(
    req: GenerateRequest, *, allow_server_path: bool
) -> tuple[str | None, Callable[[], None] | None]:
    """Resolve the reference audio to a filesystem path.

    Returns ``(path, cleanup)`` where ``cleanup`` removes the temp file (or None
    when there is nothing to clean up). ``prompt_audio_base64`` (HTTP-native upload)
    is decoded to a temp file and takes precedence; the server-side
    ``prompt_audio_path`` is only honoured when explicitly enabled in config
    (otherwise it's an arbitrary-file-read foot-gun).
    """
    if req.prompt_audio_base64:
        try:
            raw = base64.b64decode(req.prompt_audio_base64, validate=True)
        except (binascii.Error, ValueError) as e:
            raise HTTPException(status_code=400,
                                detail=f"prompt_audio_base64 is not valid base64: {e}")
        suffix = "." + req.prompt_audio_format          # schema-validated [A-Za-z0-9]{1,8}
        fd, path = tempfile.mkstemp(suffix=suffix, prefix="dots_prompt_")
        with os.fdopen(fd, "wb") as f:
            f.write(raw)

        def cleanup(p: str = path) -> None:
            with contextlib.suppress(OSError):
                os.remove(p)

        return path, cleanup

    if req.prompt_audio_path:
        if not allow_server_path:
            raise HTTPException(
                status_code=403,
                detail="server-side prompt_audio_path is disabled; send the reference "
                       "audio as prompt_audio_base64, or set DOTS_ALLOW_SERVER_AUDIO_PATH=1")
        return req.prompt_audio_path, None

    return None, None


def _gen_kwargs(req: GenerateRequest, prompt_audio_path: str | None) -> dict:
    return dict(
        num_steps=req.num_steps,
        guidance_scale=req.guidance_scale,
        eos_threshold=req.eos_threshold,
        prompt_audio_path=prompt_audio_path,
        prompt_text=req.prompt_text,
        speaker_scale=req.speaker_scale,
        clone_prefill=req.clone_prefill,
    )


@router.post("/generate", summary="Synthesize a full WAV (one-shot)")
async def generate(req: GenerateRequest, request: Request,
                   server: Any = Depends(get_server)) -> Response:
    """Full waveform as a complete WAV (one-shot vocode, high throughput)."""
    cfg = request.app.state.cfg
    path, cleanup = _resolve_prompt_audio(req, allow_server_path=cfg.allow_server_audio_path)
    try:
        wav = await server.generate_wav(req.text, **_gen_kwargs(req, path))
    finally:
        if cleanup is not None:
            cleanup()
    pcm = pcm16(wav)
    body = wav_header(server.sample_rate, len(pcm)) + pcm
    return Response(content=body, media_type="audio/wav")


@router.post("/generate_stream", summary="Stream a WAV as audio is produced")
async def generate_stream(req: GenerateRequest, request: Request,
                          server: Any = Depends(get_server)) -> StreamingResponse:
    """Stream a WAV (placeholder-length header first, then PCM16 chunks)."""
    cfg = request.app.state.cfg
    sr = int(server.sample_rate)
    path, cleanup = _resolve_prompt_audio(req, allow_server_path=cfg.allow_server_audio_path)

    async def body() -> AsyncIterator[bytes]:
        yield wav_header(sr, -1)
        async for chunk in server.generate(req.text, stream=True, **_gen_kwargs(req, path)):
            yield pcm16(chunk)

    # BackgroundTask runs after the response completes (incl. client disconnect),
    # so the temp prompt file is removed even if the stream is cut short.
    background = BackgroundTask(cleanup) if cleanup is not None else None
    return StreamingResponse(body(), media_type="audio/wav", background=background)
