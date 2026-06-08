"""nanovllm_dots.engine

Model-agnostic inference runtime core.

This package contains the components that make inference work end-to-end:

- :mod:`nanovllm_dots.engine.sequence`: per-request state machine + KV mapping.
- :mod:`nanovllm_dots.engine.block_manager`: KV-cache block pool + prefix cache.
- :mod:`nanovllm_dots.engine.scheduler`: batching policy + preemption.
- :mod:`nanovllm_dots.engine.model_runner`: GPU execution abstraction.
- :mod:`nanovllm_dots.engine.llm_engine`: orchestrates the engine step loop.

The intent is that model implementations only need to provide a thin adapter
layer ("preprocess" and "postprocess") while the runtime handles scheduling,
memory management, and execution.

Reference implementation
------------------------
For a complete, working example of how to plug a model family into this runtime,
see ``nanovllm_dots/models/voxcpm``.
"""
