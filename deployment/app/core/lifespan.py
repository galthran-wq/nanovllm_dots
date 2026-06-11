"""Application lifespan: build the DotsStreamServer once at startup.

Mirrors nano-vllm-voxcpm's ``deployment/app/core/lifespan.py``: on startup it
constructs the engine-level server (which owns the GPU engine + worker thread and
runs warmup), stashes it on ``app.state.server``, and flips ``app.state.ready``;
on shutdown it closes the server. ``DotsStreamServer.from_pretrained`` blocks until
the engine is built and warmed, so by the time the context yields the server is
ready.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from deployment.app.core.config import ServiceConfig
from deployment.app.server import DotsStreamServer

logger = logging.getLogger(__name__)


def build_lifespan(cfg: ServiceConfig):
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.ready = False
        logger.info("loading dots.tts model from %s", cfg.model_dir)
        server = DotsStreamServer.from_pretrained(
            cfg.model_dir,
            max_num_seqs=cfg.max_num_seqs,
            fm_accel=cfg.fm_accel,
            flash_pe=cfg.flash_pe,
            graph_decode=cfg.graph_decode,
            warmup=cfg.warmup,
        )
        app.state.server = server
        app.state.ready = True
        logger.info("dots.tts server ready (sample_rate=%d)", server.sample_rate)
        try:
            yield
        finally:
            app.state.ready = False
            server.close()
            if hasattr(app.state, "server"):
                delattr(app.state, "server")

    return lifespan
