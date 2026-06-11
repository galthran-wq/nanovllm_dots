"""FastAPI app for dots.tts streaming synthesis.

Thin HTTP layer over ``DotsStreamServer`` (one batched GPU engine, continuous
batching) defined in ``deployment.app.server``; that wrapper drives the
``nanovllm_dots`` GPU engine, this package only does HTTP. Mirrors the
nano-vllm-voxcpm ``deployment/`` layout.

Endpoints:
  GET  /health          -> liveness ({"status":"ok"})
  GET  /ready           -> readiness (503 until the model is loaded + warmed)
  GET  /info            -> sample_rate + service defaults
  POST /generate        -> full WAV (one-shot vocode)
  POST /generate_stream -> chunked WAV streamed as audio is produced

Run with ``scripts/serve.py`` or ``uvicorn deployment.app.main:app``.
"""
from __future__ import annotations

from fastapi import FastAPI

from deployment.app.api.api import api_router
from deployment.app.core.config import load_config
from deployment.app.core.lifespan import build_lifespan


def create_app() -> FastAPI:
    cfg = load_config()
    app = FastAPI(
        title="dots.tts",
        version="0.1.0",
        description="FastAPI wrapper for the nanovllm_dots dots.tts engine. "
                    "See /docs for interactive API docs.",
        openapi_tags=[
            {"name": "health", "description": "Liveness and readiness probes."},
            {"name": "info", "description": "Model and instance metadata."},
            {"name": "generation", "description": "Text-to-speech generation (WAV)."},
        ],
        lifespan=build_lifespan(cfg),
    )
    app.state.cfg = cfg
    app.include_router(api_router)
    return app


app = create_app()
