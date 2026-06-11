"""nanovllm_dots.engine

Model-agnostic inference runtime core.

This package contains the reusable primitives the dots engine builds on:

- :mod:`nanovllm_dots.engine.sequence`: per-request state machine + KV mapping.
- :mod:`nanovllm_dots.engine.block_manager`: KV-cache block pool + prefix cache.
- :mod:`nanovllm_dots.engine.scheduler`: batching policy + preemption.

The dots model wires these together in :mod:`nanovllm_dots.models.dots.engine`
(``DotsBatchEngine``), which owns the step loop and GPU execution rather than a
generic runner.
"""
