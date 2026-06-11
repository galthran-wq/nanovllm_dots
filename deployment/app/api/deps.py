"""FastAPI dependencies."""
from __future__ import annotations

from typing import Any, cast

from fastapi import HTTPException, Request


def get_server(request: Request) -> Any:
    """Return the DotsStreamServer, or 503 if the model is not loaded yet."""
    server = getattr(request.app.state, "server", None)
    if server is None or not getattr(request.app.state, "ready", False):
        raise HTTPException(status_code=503, detail="Model server not ready")
    # app.state is dynamically typed; normalize for type checkers.
    return cast(Any, server)
