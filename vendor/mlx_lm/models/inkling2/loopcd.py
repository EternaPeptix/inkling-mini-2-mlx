"""Opt-in LoopCD (Loop Contrastive Decoding, arXiv:2610.02185) for Inkling2 looped models.

Disabled by default (mode="off") so standard decode and prefill remain bit-exact with zero overhead.
Supports:
  * LoopCD-Hidden (Eq. 2): h' = h_R + omega * (h_R - h_ref) before coda (zero extra coda/lm_head FLOPs)
  * LoopCD-Logits (Eq. 3): z' = z_R + omega * (z_R - z_ref) via shadow coda cache + lm_head
  * Fixed omega or adaptive margin-gated omega (Eq. 4, logits mode):
      omega = omega_max * (1 - (p_{R,(1)} - p_{R,(2)}))

Configure via environment variables:
  LOOPLING_LOOPCD=hidden:0.5 | logits:0.3 | logits_adaptive:0.5 | off
  LOOPLING_LOOPCD_MODE, LOOPLING_LOOPCD_OMEGA, LOOPLING_LOOPCD_ADAPTIVE,
  LOOPLING_LOOPCD_OMEGA_MAX, LOOPLING_LOOPCD_REF, LOOPLING_LOOPCD_BLEND_BS
or programmatically via ``model.set_loopcd(...)``.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import mlx.core as mx


@dataclass
class LoopCDConfig:
    mode: str = "off"
    omega: float = 0.0
    adaptive: bool = False
    omega_max: float = 0.5
    ref_pass: int = 0
    blend_bs: bool = False

    def __post_init__(self) -> None:
        self.mode = str(self.mode).strip().lower()
        if self.mode not in ("off", "hidden", "logits"):
            raise ValueError(f"Invalid LoopCD mode {self.mode!r}; expected 'off', 'hidden', or 'logits'.")
        self.omega = float(self.omega)
        self.adaptive = bool(self.adaptive)
        self.omega_max = float(self.omega_max)
        self.ref_pass = int(self.ref_pass)
        self.blend_bs = bool(self.blend_bs)
        if self.adaptive and self.mode == "hidden":
            raise ValueError("Adaptive margin-gated strength requires mode='logits', not 'hidden'.")

    @property
    def active(self) -> bool:
        return self.mode != "off"

    @property
    def is_zero_strength(self) -> bool:
        return (self.omega_max == 0.0) if self.adaptive else (self.omega == 0.0)

    def enabled_for(self, num_passes: int) -> bool:
        return self.active and num_passes > 1

    def resolved_ref_pass(self, num_passes: int) -> int:
        idx = num_passes + self.ref_pass if self.ref_pass < 0 else self.ref_pass
        if not (0 <= idx < num_passes - 1):
            raise ValueError(
                f"LoopCD ref_pass={self.ref_pass} (resolved={idx}) out of range [0, {num_passes - 2}] "
                f"for num_passes={num_passes}."
            )
        return idx


def _parse_bool(val: str) -> bool:
    return val.strip().lower() in ("1", "true", "yes", "on")


def from_env() -> LoopCDConfig:
    spec = os.environ.get("LOOPLING_LOOPCD", "").strip()
    mode = "off"
    omega = 0.0
    adaptive = False
    omega_max = 0.5
    if spec and spec.lower() not in ("off", "0", "none"):
        parts = spec.split(":", 1)
        kind = parts[0].strip().lower()
        val = float(parts[1]) if len(parts) > 1 and parts[1].strip() != "" else None
        if kind == "hidden":
            mode = "hidden"
            omega = 0.5 if val is None else val
        elif kind == "logits":
            mode = "logits"
            adaptive = False
            omega = 0.3 if val is None else val
        elif kind in ("logits_adaptive", "adaptive"):
            mode = "logits"
            adaptive = True
            omega_max = 0.5 if val is None else val
        else:
            raise ValueError(f"Unrecognized LOOPLING_LOOPCD={spec!r}")

    if "LOOPLING_LOOPCD_MODE" in os.environ:
        mode = os.environ["LOOPLING_LOOPCD_MODE"].strip().lower()
    if "LOOPLING_LOOPCD_OMEGA" in os.environ:
        omega = float(os.environ["LOOPLING_LOOPCD_OMEGA"])
    if "LOOPLING_LOOPCD_ADAPTIVE" in os.environ:
        adaptive = _parse_bool(os.environ["LOOPLING_LOOPCD_ADAPTIVE"])
    if "LOOPLING_LOOPCD_OMEGA_MAX" in os.environ:
        omega_max = float(os.environ["LOOPLING_LOOPCD_OMEGA_MAX"])
    ref_pass = int(os.environ.get("LOOPLING_LOOPCD_REF", "0"))
    blend_bs = _parse_bool(os.environ.get("LOOPLING_LOOPCD_BLEND_BS", "0"))

    return LoopCDConfig(
        mode=mode,
        omega=omega,
        adaptive=adaptive,
        omega_max=omega_max,
        ref_pass=ref_pass,
        blend_bs=blend_bs,
    )


def apply_hidden(
    h_final: mx.array,
    bs_final: Any,
    h_ref: mx.array,
    bs_ref: Any,
    pre_n: int,
    cfg: LoopCDConfig,
) -> tuple[mx.array, Any]:
    """LoopCD-Hidden (Eq. 2): extrapolate final core hidden state away from ref_pass state."""
    if cfg.omega == 0.0:
        return h_final, bs_final
    w = float(cfg.omega)
    h_f32 = h_final.astype(mx.float32)
    h_out = (h_f32 + w * (h_f32 - h_ref.astype(mx.float32))).astype(h_final.dtype)

    if cfg.blend_bs and bs_final is not None and bs_ref is not None:
        if isinstance(bs_final, list):
            bs_out = list(bs_final[:pre_n]) + [r + w * (r - rf) for r, rf in zip(bs_final[pre_n:], bs_ref[pre_n:])]
        else:
            bs_core = bs_final[pre_n:] + w * (bs_final[pre_n:] - bs_ref[pre_n:])
            bs_out = mx.concatenate([bs_final[:pre_n], bs_core], axis=0) if pre_n > 0 else bs_core
    else:
        bs_out = bs_final
    return h_out, bs_out


def combine_logits(z_final: mx.array, z_ref: mx.array, cfg: LoopCDConfig) -> mx.array:
    """LoopCD-Logits (Eq. 3 fixed, Eq. 4 adaptive margin-gated) on f32 post-softcap logits."""
    if cfg.is_zero_strength:
        return z_final
    if not cfg.adaptive:
        w = float(cfg.omega)
        return z_final + w * (z_final - z_ref)

    # Numerically stable top-2 probabilities from log-softmax; abs() handles unsorted topk.
    lp = z_final - mx.logsumexp(z_final, axis=-1, keepdims=True)
    top2 = mx.topk(lp, 2, axis=-1)
    p_top2 = mx.exp(top2)
    margin = mx.abs(p_top2[..., 0] - p_top2[..., 1])
    w = float(cfg.omega_max) * (1.0 - margin)
    return z_final + w[..., None] * (z_final - z_ref)


def _fresh_kv_like(kv: Any, is_sliding: bool, window: int) -> Any:
    from mlx_lm.models.cache import (
        BatchKVCache,
        BatchRotatingKVCache,
        ChunkedKVCache,
        ConcatenateKVCache,
        KVCache,
        QuantizedKVCache,
        RotatingKVCache,
    )

    if isinstance(kv, BatchRotatingKVCache):
        lp = kv.left_padding.tolist() if hasattr(kv.left_padding, "tolist") else list(kv.left_padding)
        out = BatchRotatingKVCache(kv.max_size, lp)
        out._fd_delta = getattr(kv, "_fd_delta", None)
        return out
    if isinstance(kv, BatchKVCache):
        lp = kv.left_padding.tolist() if hasattr(kv.left_padding, "tolist") else list(kv.left_padding)
        out = BatchKVCache(lp)
        out._fd_delta = getattr(kv, "_fd_delta", None)
        return out
    if isinstance(kv, RotatingKVCache):
        return RotatingKVCache(max_size=kv.max_size, keep=getattr(kv, "keep", 0))
    if isinstance(kv, KVCache):
        return KVCache()
    if isinstance(kv, QuantizedKVCache):
        return QuantizedKVCache(group_size=kv.group_size, bits=kv.bits)
    if isinstance(kv, ChunkedKVCache):
        return ChunkedKVCache(kv.chunk_size)
    if isinstance(kv, ConcatenateKVCache):
        return ConcatenateKVCache()
    return RotatingKVCache(max_size=window) if is_sliding else KVCache()


def make_coda_cache_like(main_coda_caches: list[Any], coda_layers: list[Any], window: int) -> list[Any]:
    from mlx_lm.models.cache import ArraysCache, CacheList
    from mlx_lm.models.inkling2.layers import SlidingDecoderLayer

    out = []
    for mc, layer in zip(main_coda_caches, coda_layers):
        kv_new = _fresh_kv_like(mc[0], isinstance(layer, SlidingDecoderLayer), window)
        mc_conv = mc[1]
        lp = mc_conv.left_padding.tolist() if getattr(mc_conv, "left_padding", None) is not None else None
        conv_new = ArraysCache(len(mc_conv.cache), left_padding=lp)
        if getattr(mc_conv, "_zero_pad", False):
            conv_new._zero_pad = True
        out.append(CacheList(kv_new, conv_new))
    return out


def get_or_init_ref_coda(cache: list[Any], coda_layers: list[Any], window: int) -> list[Any]:
    if not coda_layers:
        return []
    if cache[0] is None:
        return [None] * len(coda_layers)
    anchor = cache[-1]
    ref_coda = getattr(anchor, "_loopcd_ref_coda", None)
    if ref_coda is None:
        main_coda = cache[-len(coda_layers) :]
        if any(not c[0].empty() for c in main_coda):
            raise RuntimeError(
                "LoopCD-Logits requires LoopCD-Logits to be enabled before prefill so the reference coda cache is populated."
            )
        ref_coda = make_coda_cache_like(main_coda, coda_layers, window)
        anchor._loopcd_ref_coda = ref_coda
    return ref_coda


def bind_ref_coda_deps(cache: list[Any], ref_coda: list[Any]) -> None:
    """Attach shadow coda arrays to main anchor cache via mx.depends so mx.eval([c.state ...]) realizes them."""
    if not cache or not ref_coda or cache[-1] is None or ref_coda[0] is None:
        return
    anchor_kv = cache[-1][0]
    if getattr(anchor_kv, "_packed", None) is not None:
        return
    ak = getattr(anchor_kv, "keys", None)
    if ak is None:
        return
    deps = []
    for rc in ref_coda:
        kv, conv = rc[0], rc[1]
        for arr in (getattr(kv, "keys", None), getattr(kv, "values", None), *conv.cache):
            if arr is not None:
                deps.append(arr)
    if deps:
        anchor_kv.keys = mx.depends(ak, deps)
