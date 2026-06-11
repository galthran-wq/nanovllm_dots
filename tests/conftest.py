"""Shared fixtures + helpers for the reference-consistency test suite.

These tests assert that the optimized dots engine (paged LLM, batched/flash/
graphed FM, flash patch_encoder, streaming vocoder, voice cloning) stays
numerically consistent with the reference ``dots_tts`` implementation. They are
all marked ``advanced``: they need a CUDA GPU and the model weights under
``models/`` -- when either is missing they skip cleanly rather than fail, so a
plain ``pytest`` on a machine without a GPU collects and skips them.

Run the whole suite:
    .venv/bin/python -m pytest tests/
Only the current (mf) path, skip legacy:
    .venv/bin/python -m pytest tests/ -m "not legacy"
One impact area:
    .venv/bin/python -m pytest tests/test_fm.py -v
"""
from __future__ import annotations

import gc
import os
import pathlib
import random
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

# --- make the project importable exactly like the scripts did (stubs/ provides
#     the `tn` stand-in; repo root provides nanovllm_dots) ---
ROOT = pathlib.Path(__file__).resolve().parent.parent
for _p in (str(ROOT / "stubs"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# Model directories (point-of-impact test files pick one via a module-level MODEL).
MF = str(ROOT / "models" / "dots.tts-mf")
SOAR = str(ROOT / "models" / "dots.tts-soar")

# Committed golden fixtures (reference latents + clone prompt audio), versioned
# alongside the tests. Regenerate with scripts/gen_golden.py.
_TESTS_DIR = pathlib.Path(__file__).resolve().parent
GOLDEN_DIR = str(_TESTS_DIR / "golden")           # soar reference latents/wavs
GOLDEN_MF_DIR = str(_TESTS_DIR / "golden_mf")     # mf reference latents/wavs

# Reference prompts (same text the golden artifacts + scripts used).
EN1 = "Hello, this is a reference sample generated for regression testing."
ZH1 = "这是一个用于回归测试的参考样本。"


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 1234) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cos_stats(a: torch.Tensor, b: torch.Tensor, *, latent_dim: int | None = None):
    """Cosine similarity per row -> (min, mean). Latents are flattened to
    ``(-1, latent_dim)`` first (the recurring idiom across the scripts)."""
    if latent_dim is not None:
        a = a.reshape(-1, latent_dim)
        b = b.reshape(-1, latent_dim)
    cos = F.cosine_similarity(a.float(), b.float(), dim=-1)
    return cos.min().item(), cos.mean().item()


def assert_cos(a, b, thresh, *, reduce="min", latent_dim=None, label=""):
    """Assert cosine similarity passes ``thresh`` (default on the per-row min)."""
    cmin, cmean = cos_stats(a, b, latent_dim=latent_dim)
    val = cmin if reduce == "min" else cmean
    assert val > thresh, f"{label} cos {reduce}={val:.6f} <= {thresh} (mean={cmean:.6f})"
    return cmin, cmean


# --------------------------------------------------------------------------- #
# model build cache (bounded to ONE model at a time to fit the shared GPU)     #
# --------------------------------------------------------------------------- #
_CUR_MODEL: str | None = None
_RUNTIME: dict[str, object] = {}
_PAGED: dict[str, object] = {}


def _evict_if_other(model_dir: str) -> None:
    global _CUR_MODEL
    if _CUR_MODEL is not None and _CUR_MODEL != model_dir:
        _RUNTIME.clear()
        _PAGED.clear()
        gc.collect()
        torch.cuda.empty_cache()
    _CUR_MODEL = model_dir


def get_runtime(model_dir: str):
    """The reference (eager) ``DotsTtsRuntime``. core/vocoder live on it."""
    _evict_if_other(model_dir)
    if model_dir not in _RUNTIME:
        from dots_tts.models.dots_tts.model import DotsTtsModel  # noqa: F401 (registers)
        from dots_tts.runtime import DotsTtsRuntime
        _RUNTIME[model_dir] = DotsTtsRuntime.from_pretrained(
            model_dir, precision="bfloat16", optimize=False, max_generate_length=256
        )
    return _RUNTIME[model_dir]


def get_paged(model_dir: str):
    """Our paged Qwen2 backbone (``QwenLLM``) loaded from the same checkpoint."""
    _evict_if_other(model_dir)
    if model_dir not in _PAGED:
        from transformers import Qwen2Config
        from nanovllm_dots.models.dots.loader import load_llm_weights
        from nanovllm_dots.models.dots.model_llm import QwenLLM
        ckpt = os.path.join(model_dir, "model.safetensors")
        cfg = Qwen2Config.from_json_file(os.path.join(model_dir, "llm_config.json"))
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("cuda"):
            paged = QwenLLM(cfg).eval()
        load_llm_weights(paged, ckpt)
        torch.set_default_dtype(torch.float32)
        _PAGED[model_dir] = paged
    return _PAGED[model_dir]


# --------------------------------------------------------------------------- #
# fixtures                                                                     #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def _nccl():
    """Init a 1-process NCCL group once (the engine/layers expect it)."""
    import torch.distributed as dist
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29600")
        dist.init_process_group("nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


@pytest.fixture(scope="module")
def model_dir(request) -> str:
    """The model a test file targets (module global ``MODEL``, default mf)."""
    md = getattr(request.module, "MODEL", MF)
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not os.path.isdir(md):
        pytest.skip(f"model weights not found: {md}")
    return md


@pytest.fixture(scope="module")
def reference(_nccl, model_dir):
    return get_runtime(model_dir)


@pytest.fixture(scope="module")
def paged(_nccl, model_dir):
    return get_paged(model_dir)


@pytest.fixture
def make_engine(_nccl, reference, paged, model_dir):
    """Factory for a (possibly subclassed) ``DotsBatchEngine``; freed after the
    test. Pass ``cls=`` to use a DualEngine shadow-path comparator subclass."""
    from nanovllm_dots.models.dots.engine import DotsBatchEngine
    built = []

    def _make(cls=DotsBatchEngine, **kw):
        kw.setdefault("num_kvcache_blocks", 256)
        kw.setdefault("block_size", 256)
        kw.setdefault("max_num_seqs", 8)
        eng = cls(reference, paged, model_dir=model_dir, **kw)
        built.append(eng)
        return eng

    yield _make
    built.clear()
    gc.collect()
    torch.cuda.empty_cache()
