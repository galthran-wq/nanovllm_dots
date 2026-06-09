#!/usr/bin/env python
"""Run the dots.tts FastAPI server.

    PYTHONPATH=stubs:. HF_HUB_OFFLINE=1 .venv/bin/python scripts/serve.py \
        --model models/dots.tts-mf --host 0.0.0.0 --port 8000

Endpoints: POST /generate, POST /generate_stream, GET /health.
One GPU / one engine / one uvicorn worker (the engine batches internally).
"""
from __future__ import annotations

import argparse
import os


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/dots.tts-mf")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    args = ap.parse_args()

    os.environ["DOTS_MODEL"] = args.model
    os.environ["DOTS_MAX_SEQS"] = str(args.max_num_seqs)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    import uvicorn
    # single worker: the engine lives in-process and batches all requests.
    uvicorn.run("nanovllm_dots.models.dots.app:app", host=args.host,
                port=args.port, workers=1, log_level="warning")


if __name__ == "__main__":
    main()
