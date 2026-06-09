#!/usr/bin/env python
"""Async server benchmark: TTFB, RTF, RPS vs concurrency (Phase 3).

Drives DotsStreamServer with N concurrent streaming generate() coroutines (one
shared batched engine) and measures, per request: time-to-first-byte (first audio
chunk) and realtime factor (wall / audio_seconds), plus aggregate throughput.
This is the real serving metric -- continuous batching with staggered async
arrivals, streamed audio, vocoder included (unlike bench_engine, latent-only).

Run:
    PYTHONPATH=stubs:. HF_HUB_OFFLINE=1 .venv/bin/python scripts/bench_server.py \
        --model models/dots.tts-mf --concurrency 1,8,32 --num-steps 4
"""
from __future__ import annotations

import argparse
import asyncio
import os
import time

import torch

EN1 = "Hello, this is a streaming synthesis benchmark for the dots engine."


async def one(server, text, num_steps, stream):
    t0 = time.perf_counter()
    ttfb = None
    samples = 0
    async for chunk in server.generate(text, num_steps=num_steps, stream=stream):
        if ttfb is None:
            ttfb = time.perf_counter() - t0
        samples += chunk.numel()
    wall = time.perf_counter() - t0
    audio_s = samples / server.sample_rate
    return dict(ttfb=ttfb or wall, wall=wall, audio_s=audio_s)


async def run_level(server, n, text, num_steps, stream):
    t0 = time.perf_counter()
    res = await asyncio.gather(*[one(server, text, num_steps, stream) for _ in range(n)])
    wall = time.perf_counter() - t0
    audio = sum(r["audio_s"] for r in res)
    ttfbs = sorted(r["ttfb"] for r in res)
    rtfs = [r["wall"] / r["audio_s"] if r["audio_s"] else 0 for r in res]
    return dict(
        n=n, wall=wall, audio=audio, rps=n / wall, thrpt=audio / wall,
        ttfb_med=ttfbs[len(ttfbs) // 2], ttfb_p95=ttfbs[min(len(ttfbs) - 1, int(0.95 * len(ttfbs)))],
        rtf_med=sorted(rtfs)[len(rtfs) // 2], rtf_max=max(rtfs),
    )


async def main_async(args):
    from nanovllm_dots.models.dots.server import DotsStreamServer
    server = DotsStreamServer.from_pretrained(
        args.model, max_num_seqs=args.max_num_seqs)
    server.vocode = not args.no_vocode

    # warm the engine graphs with a single request (first one captures shapes)
    await server.generate_wav(EN1, num_steps=args.num_steps)

    if args.sync:
        # diagnostic: drive the engine's step_stream directly (no worker thread, no
        # asyncio) to isolate engine throughput from the async/thread machinery.
        import time as _t
        server._stop.set(); server._thread.join(timeout=5)
        eng = server.engine
        print(f"mode=SYNC stream={args.stream} vocode={not args.no_vocode}")
        print(f"{'conc':>5} {'wall_s':>7} {'audio_s':>8} {'thrpt(xRT)':>10}")
        def drive(n, tag):
            for i in range(n):
                eng.add_request(f"{tag}{n}_{i}", EN1, num_steps=args.num_steps,
                                stream=args.stream,
                                eos_threshold=2.0 if args.fixed_len else 0.8,
                                max_patches=args.fixed_len or None)
            torch.cuda.synchronize(); t0 = _t.perf_counter()
            samples = 0
            while not eng.scheduler.is_finished():
                for sid, kind, data in eng.step_stream(vocode=not args.no_vocode):
                    if kind == "audio":
                        samples += data.numel()
            torch.cuda.synchronize()
            return _t.perf_counter() - t0, samples / server.sample_rate

        for n in [int(x) for x in args.concurrency.split(",")]:
            drive(n, "warm")                       # capture graphs for this batch size
            wall, audio = drive(n, "meas")
            print(f"{n:>5} {wall:>7.2f} {audio:>8.1f} {audio / wall:>10.1f}")
        return

    print(f"mode={'stream' if args.stream else 'oneshot'} "
          f"vocode={not args.no_vocode}")
    print(f"{'conc':>5} {'wall_s':>7} {'audio_s':>8} {'thrpt(xRT)':>10} "
          f"{'RPS':>6} {'TTFB_med':>9} {'TTFB_p95':>9} {'RTF_med':>8} {'RTF_max':>8}")
    for n in [int(x) for x in args.concurrency.split(",")]:
        r = await run_level(server, n, EN1, args.num_steps, args.stream)
        print(f"{r['n']:>5} {r['wall']:>7.2f} {r['audio']:>8.1f} {r['thrpt']:>10.1f} "
              f"{r['rps']:>6.2f} {r['ttfb_med']:>9.3f} {r['ttfb_p95']:>9.3f} "
              f"{r['rtf_med']:>8.3f} {r['rtf_max']:>8.3f}")
    server.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/dots.tts-mf")
    ap.add_argument("--concurrency", default="1,8,32")
    ap.add_argument("--num-steps", type=int, default=4)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--no-vocode", action="store_true",
                    help="emit silent chunks (isolate vocoder cost)")
    ap.add_argument("--stream", action="store_true",
                    help="per-patch streaming vocode (default: one-shot at end)")
    ap.add_argument("--sync", action="store_true",
                    help="diagnostic: drive engine.step_stream directly (no async)")
    ap.add_argument("--fixed-len", type=int, default=0,
                    help="(sync only) force exactly N patches/req (eos off)")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
