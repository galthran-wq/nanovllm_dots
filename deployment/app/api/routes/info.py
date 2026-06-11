from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request

from deployment.app.api.deps import get_server
from deployment.app.schemas.tts import InfoResponse

router = APIRouter(tags=["info"])


@router.get(
    "/info",
    response_model=InfoResponse,
    summary="Model and service metadata",
    responses={503: {"description": "Model server not ready"}},
)
async def info(request: Request, server: Any = Depends(get_server)) -> InfoResponse:
    cfg = request.app.state.cfg
    return InfoResponse(
        sample_rate=int(server.sample_rate),
        model_dir=cfg.model_dir,
        max_num_seqs=cfg.max_num_seqs,
        default_num_steps=cfg.default_num_steps,
        default_guidance_scale=cfg.default_guidance_scale,
        default_eos_threshold=cfg.default_eos_threshold,
    )
