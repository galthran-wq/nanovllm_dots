from __future__ import annotations

from fastapi import APIRouter

from deployment.app.api.routes.generate import router as generate_router
from deployment.app.api.routes.health import router as health_router
from deployment.app.api.routes.info import router as info_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(info_router)
api_router.include_router(generate_router)
