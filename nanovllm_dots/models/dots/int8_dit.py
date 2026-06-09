"""W8A8 dynamic int8 quantization for the DiT linears (exploratory).

A100 (sm_80) has int8 tensor cores: an int8xint8->int32 GEMM runs ~2x the bf16
rate. Under concurrency the batched FM (DiT forward) is the compute-bound stage
(~66% of wall), so quantizing its big linears (75x 1024^2, 18x ffn 4096, 18x
adaLN 6144) could raise the throughput ceiling -- IF the accuracy cost is
acceptable.

This is per-token-dynamic activation + per-out-channel-static weight quant
(the standard W8A8 recipe), built directly on `torch._int_mm` so it needs no
torchao. Activation scales are computed per forward (data-dependent but
fixed-shape, so still CUDA-graph capturable). Faithfulness is NOT exact -- this
module exists to MEASURE the speed/quality tradeoff, not to ship.

RESULT (mf, NFE=4, flash_pe, fixed-len 20, A100, conc=64) -- NEGATIVE, ~2x SLOWER:
  cudagraph: bf16 57.8x RT  -> int8 27.8x RT     (single-stream 0.275 -> 0.612)
  compile:   bf16 60.7x RT  -> int8 30.1x RT
Quality is fine (per-linear cos 0.99995, ~1% rel err) -- the loss is pure speed.
Why: the DiT GEMMs are small (1024^2, 1024x4096), so bf16 tensor cores are
already fast and int8's theoretical 2x is tiny in absolute terms; meanwhile the
unfused quant/dequant (per-token amax, round, int8 cast, int32->fp32 dequant with
two broadcasts on [M,out]) adds memory traffic comparable to the GEMM itself.
inductor does NOT fuse the torch._int_mm dequant epilogue (compile+int8 also 2x
slower), and under cudagraph the extra compute is hidden-launch but still real.
int8 would need fused CUTLASS kernels (torchao) AND larger matrices to win; on a
1024-dim DiT the upside is marginal even fused. Kept as a documented dead-end.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Int8DynamicLinear(nn.Module):
    """Drop-in for nn.Linear: int8 weight (per-out-channel) x int8 activation
    (per-token, dynamic) via torch._int_mm, dequantized back to the input dtype."""

    def __init__(self, lin: nn.Linear):
        super().__init__()
        W = lin.weight.data.float()                       # [out, in]
        w_scale = (W.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-8)
        W_int8 = torch.round(W / w_scale).clamp(-127, 127).to(torch.int8)
        # store the GEMM rhs as [in, out] contiguous int8 (b of _int_mm)
        self.register_buffer("Wt_int8", W_int8.t().contiguous())
        self.register_buffer("w_scale", w_scale.squeeze(1).float())   # [out]
        self.bias = lin.bias                              # keep (bf16) or None
        self.in_features = lin.in_features
        self.out_features = lin.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2 = x.reshape(-1, self.in_features)
        M = x2.size(0)
        pad = 17 - M if M < 17 else 0          # _int_mm requires m > 16
        if pad:
            x2 = torch.nn.functional.pad(x2, (0, 0, 0, pad))
        x_scale = (x2.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-8)  # [M,1]
        x_int8 = torch.round(x2 / x_scale).clamp(-127, 127).to(torch.int8)
        acc = torch._int_mm(x_int8, self.Wt_int8)          # [M, out] int32
        out = acc.float() * x_scale * self.w_scale[None, :]
        if self.bias is not None:
            out = out + self.bias.float()
        if pad:
            out = out[:M]
        return out.to(x.dtype).reshape(*shape[:-1], self.out_features)


# the small/sensitive linears we leave in bf16: input/output projections and the
# adaLN of tiny in-features. Everything else (the per-block attn/ffn/adaLN GEMMs)
# is quantized.
def quantize_dit_(dit, *, min_features: int = 512) -> int:
    """Replace eligible nn.Linear in `dit` with Int8DynamicLinear, in place.
    Skips linears whose in/out is < min_features (tiny, latency-irrelevant, and
    the in/out projections that carry the signal in/out of the residual stream).
    Returns the count quantized. _int_mm needs in/out multiples of ~8, which all
    eligible DiT linears satisfy (1024/4096/6144)."""
    n = 0
    for name, mod in dit.named_modules():
        for child_name, child in list(mod.named_children()):
            if isinstance(child, nn.Linear) and not isinstance(child, Int8DynamicLinear):
                if min(child.in_features, child.out_features) < min_features:
                    continue
                setattr(mod, child_name, Int8DynamicLinear(child).to(child.weight.device))
                n += 1
    return n
