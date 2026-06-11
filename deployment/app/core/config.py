"""Service configuration for the dots.tts FastAPI deployment.

All knobs come from environment variables (read once at startup, mirroring the
nano-vllm-voxcpm deployment). The dots engine itself is configured through
``DotsStreamServer.from_pretrained`` -- this just collects the values.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v else default


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v else default


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    return float(v) if v else default


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class ServiceConfig:
    # model / engine
    model_dir: str = "models/dots.tts-mf"
    max_num_seqs: int = 64
    fm_accel: str = "compile"           # "compile" | "cudagraph"
    flash_pe: bool = True
    graph_decode: bool = True
    warmup: bool = True
    # serving
    host: str = "0.0.0.0"
    port: int = 8000
    # security: allow a request to point prompt_audio_path at a server-side file.
    # Off by default -- otherwise any caller could read arbitrary paths. base64
    # upload (prompt_audio_base64) always works regardless of this flag.
    allow_server_audio_path: bool = False
    # request defaults (overridable per-request)
    default_num_steps: int = 4
    default_guidance_scale: float = 1.2
    default_eos_threshold: float = 0.8


def load_config() -> ServiceConfig:
    """Build the service config from environment variables.

    Recognised vars: DOTS_MODEL, DOTS_MAX_SEQS, DOTS_FM_ACCEL, DOTS_FLASH_PE,
    DOTS_GRAPH_DECODE, DOTS_WARMUP, DOTS_HOST, DOTS_PORT, DOTS_NUM_STEPS,
    DOTS_GUIDANCE_SCALE, DOTS_EOS_THRESHOLD.
    """
    return ServiceConfig(
        model_dir=_env_str("DOTS_MODEL", "models/dots.tts-mf"),
        max_num_seqs=_env_int("DOTS_MAX_SEQS", 64),
        fm_accel=_env_str("DOTS_FM_ACCEL", "compile"),
        flash_pe=_env_bool("DOTS_FLASH_PE", True),
        graph_decode=_env_bool("DOTS_GRAPH_DECODE", True),
        warmup=_env_bool("DOTS_WARMUP", True),
        host=_env_str("DOTS_HOST", "0.0.0.0"),
        port=_env_int("DOTS_PORT", 8000),
        allow_server_audio_path=_env_bool("DOTS_ALLOW_SERVER_AUDIO_PATH", False),
        default_num_steps=_env_int("DOTS_NUM_STEPS", 4),
        default_guidance_scale=_env_float("DOTS_GUIDANCE_SCALE", 1.2),
        default_eos_threshold=_env_float("DOTS_EOS_THRESHOLD", 0.8),
    )
