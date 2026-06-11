"""Request/response schemas for the dots.tts HTTP API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class InfoResponse(BaseModel):
    sample_rate: int
    model_dir: str
    max_num_seqs: int
    default_num_steps: int
    default_guidance_scale: float
    default_eos_threshold: float


class GenerateRequest(BaseModel):
    """A TTS request.

    Voice cloning (optional) accepts the reference audio two ways:
      - ``prompt_audio_base64`` (+ ``prompt_audio_format``): the audio file bytes,
        base64-encoded in the JSON body -- the HTTP-native way, no server-side file
        needed.
      - ``prompt_audio_path``: a path to a wav on the *server* filesystem (handy for
        baked-in reference voices / local testing).
    ``prompt_text`` is the reference transcript; supplying it enables the stronger
    in-context prefill clone (``clone_prefill=True``). With no prompt audio the
    request is plain (un-cloned) synthesis.
    """

    text: str = Field(..., min_length=1)
    num_steps: int = Field(4, ge=1, le=64)
    guidance_scale: float = Field(1.2, ge=0.0)
    eos_threshold: float = Field(0.8, ge=0.0)

    # voice cloning
    prompt_audio_base64: str | None = None
    # bare audio-file extension (no dot/separators) -- used only as a temp-file suffix.
    prompt_audio_format: str = Field("wav", pattern=r"^[A-Za-z0-9]{1,8}$")
    prompt_audio_path: str | None = None
    prompt_text: str | None = None
    speaker_scale: float = 1.5
    clone_prefill: bool = True
