# Copyright © 2026 Apple Inc.

import math
import os
from functools import partial
from typing import Any

import mlx.core as mx
from mlx import nn
from mlx.nn.layers.distributed import sum_gradients

from mlx_lm.kernels.expert_routing import expert_routing
from mlx_lm.layers.moe import SwitchLinear, _gather_sort, _scatter_unsort
from mlx_lm.models.cache import BatchRotatingKVCache, RotatingKVCache
from mlx_lm.models.inkling2.config import TextArgs
from mlx_lm.models.inkling2.kernels import depth_attention, relative_attention
from mlx_lm.models.inkling2 import fastdecode
from mlx_lm.models.inkling2 import prefill as P

__all__ = [
    "Attention",
    "BlockAttentionResidual",
    "DecoderLayer",
    "DynamicShortConvolution",
    "Experts",
    "FullAttention",
    "FullDecoderLayer",
    "L2Clip",
    "MessageGate",
    "MessageInput",
    "RelativeLogits",
    "ScaledL2Clip",
    "ShortConvolution",
    "SlidingAttention",
    "SlidingDecoderLayer",
    "SoftcappedSwiGLU",
    "SparseMoeBlock",
    "TopKRouter",
    "unweighted_rms_norm",
]


_ROUTER_SCORE_EPS = 1e-20
_REF_ATTN = bool(__import__("os").environ.get("LOOPLING_REF_ATTN"))  # diagnostics: multi-token attention via the reference path
_FAST_ON = os.environ.get("LOOPLING_COMPILE_DECODE", "1") not in ("0", "")  # default ON; =0 disables


def _use_kernels() -> bool:
    return mx.default_device() == mx.gpu and mx.metal.is_available()


def unweighted_rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    return mx.fast.rms_norm(x, None, eps)


@partial(mx.compile, shapeless=True)
def _l2_clip(hidden_states: mx.array, scale: mx.array, threshold: float) -> mx.array:
    states = hidden_states.astype(mx.float32) * scale
    norm = mx.sqrt((states * states).sum(axis=-1, keepdims=True))
    scaled = states * mx.minimum(threshold / (norm + 1e-6), 1.0)
    return scaled.astype(hidden_states.dtype)


@partial(mx.compile, shapeless=True)
def _softcapped_swiglu(
    gate: mx.array, up: mx.array, gate_softcap: float, up_softcap: float
) -> mx.array:
    input_dtype = gate.dtype
    gate, up = gate.astype(mx.float32), up.astype(mx.float32)
    gate_capped = mx.tanh(gate / gate_softcap) * gate_softcap
    up_capped = mx.tanh(up / up_softcap) * up_softcap
    return (gate_capped * mx.sigmoid(gate) * up_capped).astype(input_dtype)


@partial(mx.compile, shapeless=True)
def _attend_post_gate(attn_output: mx.array, gate: mx.array) -> mx.array:
    out = mx.flatten(attn_output, 2).astype(mx.float32)
    return (out * mx.sigmoid(gate.astype(mx.float32))).astype(gate.dtype)


class L2Clip(nn.Module):
    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return _l2_clip(hidden_states, 1.0, self.threshold)


class ScaledL2Clip(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, threshold: float):
        super().__init__()
        self.threshold = threshold
        self.scale = mx.ones((num_heads, head_dim))

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return _l2_clip(hidden_states, self.scale, self.threshold)


class RelativeLogits(nn.Module):
    def __init__(self, d_rel: int, rel_extent: int):
        super().__init__()
        self.proj = mx.zeros((d_rel, rel_extent))

    def __call__(self, relative_states: mx.array) -> mx.array:
        dt = relative_states.dtype
        p = self.proj
        if p.dtype != dt:
            c = getattr(self, "_proj_cache", None)
            if c is None or c[0] is not p or c[1] != dt:
                c = (p, dt, p.astype(dt))
                self._proj_cache = c
            p = c[2]
        return relative_states @ p


def _conv_input(
    hidden_states: mx.array,
    conv_state: mx.array | None,
    conv_mask: mx.array | None,
    kernel_size: int,
) -> mx.array:
    hidden_states = hidden_states.astype(mx.float32)
    if conv_mask is not None:
        hidden_states = mx.where(conv_mask[..., None], hidden_states, 0)
    if conv_state is None:
        shape = (hidden_states.shape[0], kernel_size - 1, hidden_states.shape[-1])
        conv_state = mx.zeros(shape, mx.float32)
    return mx.concatenate([conv_state, hidden_states], axis=1)


def _conv_state(window: mx.array, lengths: mx.array | None, n: int) -> mx.array:
    if lengths is None:
        pos = mx.arange(window.shape[1] - n, window.shape[1])[None, :, None]
    else:
        ends = mx.clip(lengths, 0, window.shape[1] - n)
        pos = (ends[:, None] + mx.arange(n))[..., None]
    # A gather copies, so the state does not keep the input in memory.
    return mx.take_along_axis(window, pos, axis=1)


class ShortConvolution(nn.Module):
    def __init__(self, dims: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv1d = nn.Conv1d(dims, dims, kernel_size, groups=dims, bias=False)

    def __call__(
        self,
        hidden_states: mx.array,
        conv_state: mx.array | None,
        conv_mask: mx.array | None,
        *,
        fast: bool = False,
        conv_lengths: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        K = self.kernel_size
        if (
            fast
            and K == 4
            and P.enabled(hidden_states)
            and self.conv1d.weight.dtype == mx.bfloat16
            and (conv_state is None or conv_state.dtype == mx.float32)
        ):
            return P.short_conv(
                hidden_states,
                conv_state,
                self.conv1d.weight,
                conv_mask=conv_mask,
                conv_lengths=conv_lengths,
            )
        residual = hidden_states.astype(mx.float32)
        window = _conv_input(hidden_states, conv_state, conv_mask, K)
        # As the reference, the conv runs in the weight dtype.
        out = self.conv1d(window.astype(self.conv1d.weight.dtype))
        out = (out.astype(mx.float32) + residual) / math.sqrt(K + 1)
        return out.astype(hidden_states.dtype), window


class DynamicShortConvolution(nn.Module):
    def __init__(self, dims: int, input_dims: int, num_heads: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.static_weight = mx.zeros((kernel_size, dims))
        self.weight_proj = nn.Linear(input_dims, num_heads * kernel_size, bias=False)

    def __call__(
        self,
        hidden_states: mx.array,
        dynamic_input: mx.array,
        conv_state: mx.array | None,
        conv_mask: mx.array | None,
        *,
        fast: bool = False,
        conv_lengths: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        B, L, C = hidden_states.shape
        K, H = self.kernel_size, self.num_heads
        if (
            fast
            and K == 4
            and P.enabled(hidden_states)
            and self.static_weight.dtype == mx.bfloat16
            and (conv_state is None or conv_state.dtype == mx.float32)
        ):
            dw = self.weight_proj(dynamic_input)
            if dw.dtype == mx.bfloat16:
                return P.dyn_short_conv(
                    hidden_states,
                    dw,
                    self.static_weight,
                    conv_state,
                    H,
                    conv_mask=conv_mask,
                    conv_lengths=conv_lengths,
                )
            dynamic_weights = dw.astype(mx.float32).reshape(B, L, H, 1, K)
        else:
            dynamic_weights = self.weight_proj(dynamic_input).astype(mx.float32).reshape(B, L, H, 1, K)
        static_weights = self.static_weight.astype(mx.float32).T.reshape(H, -1, K)
        weights = static_weights + dynamic_weights

        window = _conv_input(hidden_states, conv_state, conv_mask, K)
        windows = mx.stack(
            [window[:, K - 1 - lag : K - 1 - lag + L] for lag in range(K)], axis=-1
        )
        windows = windows.reshape(B, L, H, -1, K)
        out = (windows * weights).sum(axis=-1).reshape(B, L, C)
        out = (out + hidden_states.astype(mx.float32)) / math.sqrt(2 * K + 1)
        return out.astype(hidden_states.dtype), window


def _positions(offset, length: int) -> mx.array:
    pos = mx.arange(length)
    if isinstance(offset, mx.array):
        return offset[:, None] + pos
    return pos + offset


def _key_positions(cache: Any, num_keys: int, num_queries: int, end) -> mx.array:
    k = mx.arange(num_keys)
    if num_queries == 1 and isinstance(cache, (RotatingKVCache, BatchRotatingKVCache)):
        # A decode step writes in place, so the keys can be rotated.
        k = (k - cache._idx) % num_keys
    if isinstance(end, mx.array):
        return (end - num_keys)[:, None] + k
    return k + (end - num_keys)


def _attention_reference(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    position_bias: mx.array,
    sinks: mx.array,
    *,
    scale: float,
    window: int,
    query_positions: mx.array,
    key_positions: mx.array,
) -> mx.array:
    B, Hq, L, D = query.shape
    Hkv, S = key.shape[1], key.shape[2]
    R = position_bias.shape[-1]
    distance = query_positions[..., :, None] - key_positions[..., None, :]
    visible = (distance >= 0) & (key_positions[..., None, :] >= 0)
    if window:
        visible = visible & (distance < window)
    if distance.ndim == 3:
        distance, visible = distance[:, None], visible[:, None]
    bias = mx.take_along_axis(
        position_bias.transpose(0, 2, 1, 3),
        mx.broadcast_to(mx.clip(distance, 0, R - 1), (B, Hq, L, S)),
        axis=-1,
    )
    bias = mx.where(distance < R, bias.astype(mx.float32), 0.0)
    attention_mask = mx.where(visible, bias, -mx.inf).reshape(B, Hkv, -1, L, S)

    query = query.astype(mx.float32).reshape(B, Hkv, -1, L, D)
    key = key.astype(mx.float32)[:, :, None]
    attn_weights = (query @ key.swapaxes(-1, -2)) * scale + attention_mask
    sinks = sinks.astype(mx.float32).transpose(0, 2, 1).reshape(B, Hkv, -1, L, 1)
    combined_logits = mx.concatenate([attn_weights, sinks], axis=-1)
    attn_weights = mx.softmax(combined_logits, axis=-1)[..., :S]
    attn_output = attn_weights.astype(value.dtype) @ value[:, :, None]
    return attn_output.reshape(B, Hq, L, D).transpose(0, 2, 1, 3)


class Attention(nn.Module):
    def __init__(self, args: TextArgs, *, sliding: bool):
        super().__init__()
        if sliding:
            self.num_heads = args.swa_num_attention_heads
            self.num_key_value_heads = args.swa_num_key_value_heads
            self.head_dim = args.swa_head_dim
            self.sliding_window = args.sliding_window_size
            rel_extent = args.sliding_window_size
        else:
            self.num_heads = args.num_attention_heads
            self.num_key_value_heads = args.num_key_value_heads
            self.head_dim = args.head_dim
            self.sliding_window = 0
            rel_extent = args.rel_extent
        self.scaling = self.head_dim**-0.5

        dim = args.hidden_size
        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_key_value_heads * self.head_dim
        self.q_proj = nn.Linear(dim, q_dim, bias=False)
        self.k_proj = nn.Linear(dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(dim, kv_dim, bias=False)
        self.r_proj = nn.Linear(dim, self.num_heads * args.d_rel, bias=False)
        self.o_proj = nn.Linear(q_dim, dim, bias=False)
        self.gate_proj = nn.Linear(dim, q_dim, bias=False)
        self.sink_proj = nn.Linear(dim, self.num_heads, bias=False)
        self.rel_logits_proj = RelativeLogits(args.d_rel, rel_extent)
        self.q_l2_clip = ScaledL2Clip(
            self.num_heads, self.head_dim, args.qk_l2_clip_threshold
        )
        self.k_l2_clip = L2Clip(args.qk_l2_clip_threshold)
        self.rel_l2_clip = ScaledL2Clip(
            self.num_heads, args.d_rel, args.rel_l2_clip_threshold
        )
        self.eps = args.rms_norm_eps

    def _rel_proj_bf16(self, dt: mx.Dtype) -> mx.array:
        p = self.rel_logits_proj.proj
        c = getattr(self.rel_logits_proj, "_proj_cache", None)
        if c is None or c[0] is not p or c[1] != dt:
            p_dt = mx.contiguous(p.astype(dt) if p.dtype != dt else p)
            c = (p, dt, p_dt)
            self.rel_logits_proj._proj_cache = c
        return c[2]

    def _otf_rel_ok(self, hidden_states: mx.array) -> bool:
        return (
            hidden_states.shape[1] > 1
            and P._PF_OTF_REL
            and _use_kernels()
            and not _REF_ATTN
            and hidden_states.dtype == mx.bfloat16
            and self.rel_l2_clip.scale.dtype == mx.bfloat16
            and self.rel_logits_proj.proj.shape[0] == 16
            and self.rel_logits_proj.proj.shape[1] >= 24
        )

    def _attend_pre(
        self,
        hidden_states: mx.array,
        query_states: mx.array,
        key_states: mx.array,
        value_states: mx.array,
        relative_states: mx.array,
        tau: mx.array | None = None,
        gate: mx.array | None = None,
        sinks: mx.array | None = None,
        max_dist: int | None = None,
        otf_rel: bool = False,
    ):
        B, L, _ = hidden_states.shape
        if gate is None:
            gate = P.aff8_linear(self.gate_proj, hidden_states) if P._PF_AFF8_QMM and P.enabled(hidden_states) else self.gate_proj(hidden_states)
            sinks = self.sink_proj(hidden_states)

        if (
            P._PF_ATTEND_PRE
            and P.enabled(hidden_states)
            and self.head_dim == 128
            and query_states.dtype == mx.bfloat16
            and key_states.dtype == mx.bfloat16
            and value_states.dtype == mx.bfloat16
            and self.q_l2_clip.scale.dtype == mx.bfloat16
        ):
            if otf_rel and self.rel_l2_clip.scale.dtype == mx.bfloat16 and relative_states.shape[-1] == 16 * self.num_heads:
                query_states, key_states, value_states, position_bias = P.qkv_clip_rms(
                    query_states,
                    key_states,
                    value_states,
                    self.q_l2_clip.scale,
                    num_heads=self.num_heads,
                    num_kv_heads=self.num_key_value_heads,
                    thr_q=self.q_l2_clip.threshold,
                    thr_k=self.k_l2_clip.threshold,
                    eps=self.eps,
                    tau=tau,
                    r=relative_states,
                    r_scale=self.rel_l2_clip.scale,
                    thr_r=self.rel_l2_clip.threshold,
                )
                return query_states, key_states, value_states, position_bias, sinks, gate
            query_states, key_states, value_states = P.qkv_clip_rms(
                query_states,
                key_states,
                value_states,
                self.q_l2_clip.scale,
                num_heads=self.num_heads,
                num_kv_heads=self.num_key_value_heads,
                thr_q=self.q_l2_clip.threshold,
                thr_k=self.k_l2_clip.threshold,
                eps=self.eps,
                tau=tau,
            )
            relative_states = self.rel_l2_clip(relative_states.reshape(B, L, self.num_heads, -1))
            if tau is not None:
                relative_states = (relative_states.astype(mx.float32) * tau).astype(
                    relative_states.dtype
                )
            if otf_rel:
                return query_states, key_states, value_states, relative_states, sinks, gate
            md = ((max_dist + 63) // 64) * 64 if max_dist is not None else None
            if md is not None and 0 < md < self.rel_logits_proj.proj.shape[1]:
                p = self._rel_proj_bf16(relative_states.dtype)
                position_bias = relative_states @ p[:, :md]
            else:
                position_bias = self.rel_logits_proj(relative_states)
            return query_states, key_states, value_states, position_bias, sinks, gate

        query_states = query_states.reshape(B, L, self.num_heads, -1)
        key_states = key_states.reshape(B, L, self.num_key_value_heads, -1)
        value_states = value_states.reshape(B, L, self.num_key_value_heads, -1)
        relative_states = relative_states.reshape(B, L, self.num_heads, -1)
        query_states = self.q_l2_clip(query_states)
        key_states = self.k_l2_clip(key_states)
        value_states = unweighted_rms_norm(
            value_states.astype(mx.float32), self.eps
        ).astype(value_states.dtype)
        relative_states = self.rel_l2_clip(relative_states)
        if tau is not None:
            query_states = (query_states.astype(mx.float32) * tau).astype(
                query_states.dtype
            )
            relative_states = (relative_states.astype(mx.float32) * tau).astype(
                relative_states.dtype
            )
        position_bias = self.rel_logits_proj(relative_states)

        query_states = query_states.transpose(0, 2, 1, 3)
        key_states = key_states.transpose(0, 2, 1, 3)
        value_states = value_states.transpose(0, 2, 1, 3)
        return query_states, key_states, value_states, position_bias, sinks, gate

    def _attend_post(self, attn_output: mx.array, gate: mx.array) -> mx.array:
        B, L = gate.shape[0], gate.shape[1]
        if P.enabled(gate):
            x = _attend_post_gate(attn_output, gate)
            return P.aff8_linear(self.o_proj, x) if P._PF_AFF8_QMM else self.o_proj(x)
        attn_output = attn_output.reshape(B, L, -1).astype(mx.float32)
        attn_output = attn_output * mx.sigmoid(gate.astype(mx.float32))
        return self.o_proj(attn_output.astype(gate.dtype))

    def _attend(
        self,
        hidden_states: mx.array,
        query_states: mx.array,
        key_states: mx.array,
        value_states: mx.array,
        relative_states: mx.array,
        cache: Any | None,
        tau: mx.array | None = None,
    ) -> mx.array:
        max_dist = None
        if cache is None:
            max_dist = hidden_states.shape[1]
        elif isinstance(cache.offset, int):
            max_dist = cache.offset + hidden_states.shape[1]
        otf_rel = self._otf_rel_ok(hidden_states)
        query_states, key_states, value_states, position_bias, sinks, gate = (
            self._attend_pre(
                hidden_states,
                query_states,
                key_states,
                value_states,
                relative_states,
                tau,
                max_dist=max_dist,
                otf_rel=otf_rel,
            )
        )
        if cache is not None:
            key_states, value_states = cache.update_and_fetch(key_states, value_states)
        rel_proj = self._rel_proj_bf16(hidden_states.dtype) if otf_rel else None
        attn_output = self._attention(
            query_states, key_states, value_states, position_bias, sinks, cache, rel_proj=rel_proj
        )
        return self._attend_post(attn_output, gate)

    def _attention(self, query, key, value, position_bias, sinks, cache, rel_proj=None) -> mx.array:
        B, _, L, _ = query.shape
        S = key.shape[2]
        end = cache.offset if cache is not None else L
        bf16 = query.dtype == key.dtype == value.dtype == mx.bfloat16
        if L > 1 and _use_kernels() and bf16 and not _REF_ATTN:
            key_start = mx.broadcast_to(S - end, (B,))
            return relative_attention(
                query,
                key,
                value,
                position_bias,
                sinks,
                scale=self.scaling,
                window=self.sliding_window,
                key_start=key_start,
                rel_proj=rel_proj,
            )

        query_positions = _positions(end - L, L)
        key_positions = _key_positions(cache, S, L, end)
        if L == 1 and B == 1:
            # The fused attention takes the sinks of one row.
            distance = query_positions.reshape(1) - key_positions.reshape(S)
            R = position_bias.shape[-1]
            bias = mx.take(position_bias[0, 0], mx.clip(distance, 0, R - 1), axis=-1)
            bias = mx.where(distance < R, bias, mx.zeros_like(bias))
            visible = (distance >= 0) & (key_positions.reshape(S) >= 0)
            if self.sliding_window:
                visible = visible & (distance < self.sliding_window)
            mask = mx.where(visible, bias, -mx.inf).astype(query.dtype)
            attn_output = mx.fast.scaled_dot_product_attention(
                query,
                key,
                value,
                scale=self.scaling,
                mask=mask[None, :, None, :],
                sinks=sinks[0, 0].astype(query.dtype),
            )
            return attn_output.transpose(0, 2, 1, 3)

        return _attention_reference(
            query,
            key,
            value,
            position_bias,
            sinks,
            scale=self.scaling,
            window=self.sliding_window,
            query_positions=query_positions,
            key_positions=key_positions,
        )


class SlidingAttention(Attention):
    def __init__(self, args: TextArgs):
        super().__init__(args, sliding=True)
        dims = self.num_key_value_heads * self.head_dim
        self.k_sconv = ShortConvolution(dims, args.sconv_kernel_size)
        self.v_sconv = ShortConvolution(dims, args.sconv_kernel_size)

    def __call__(
        self,
        hidden_states: mx.array,
        key_hidden_states: mx.array,
        value_hidden_states: mx.array,
        *,
        cache: Any | None,
        conv_states: list,
        conv_mask: mx.array | None,
        fast_conv: bool = False,
        conv_lengths: mx.array | None = None,
    ) -> mx.array:
        kv_cache = cache[0] if cache is not None else None
        use_a8 = P._PF_AFF8_QMM and P.enabled(hidden_states)
        if (
            fast_conv
            and P._PF_ATTEND_PRE
            and P.enabled(hidden_states)
            and self.head_dim == 128
            and self.k_sconv.kernel_size == 4
            and self.v_sconv.kernel_size == 4
            and self.k_sconv.conv1d.weight.dtype == mx.bfloat16
            and self.v_sconv.conv1d.weight.dtype == mx.bfloat16
            and self.q_l2_clip.scale.dtype == mx.bfloat16
            and (conv_states[0] is None or conv_states[0].dtype == mx.float32)
            and (conv_states[1] is None or conv_states[1].dtype == mx.float32)
        ):
            B, L, _ = hidden_states.shape
            q_raw = P.aff8_linear(self.q_proj, hidden_states) if use_a8 else self.q_proj(hidden_states)
            k_raw = P.aff8_linear(self.k_proj, key_hidden_states) if use_a8 else self.k_proj(key_hidden_states)
            v_raw = P.aff8_linear(self.v_proj, value_hidden_states) if use_a8 else self.v_proj(value_hidden_states)
            r_raw = P.aff8_linear(self.r_proj, hidden_states) if use_a8 else self.r_proj(hidden_states)
            gate = P.aff8_linear(self.gate_proj, hidden_states) if use_a8 else self.gate_proj(hidden_states)
            sinks = self.sink_proj(hidden_states)
            if self._otf_rel_ok(hidden_states):
                query_states, key_states, value_states, conv_states[0], conv_states[1], position_bias = P.swa_qkv_conv_clip(
                    q_raw,
                    k_raw,
                    v_raw,
                    self.q_l2_clip.scale,
                    self.k_sconv.conv1d.weight,
                    self.v_sconv.conv1d.weight,
                    conv_states[0],
                    conv_states[1],
                    num_heads=self.num_heads,
                    num_kv_heads=self.num_key_value_heads,
                    thr_q=self.q_l2_clip.threshold,
                    thr_k=self.k_l2_clip.threshold,
                    eps=self.eps,
                    r=r_raw,
                    r_scale=self.rel_l2_clip.scale,
                    thr_r=self.rel_l2_clip.threshold,
                    conv_mask=conv_mask,
                    conv_lengths=conv_lengths,
                )
                rel_proj = self._rel_proj_bf16(hidden_states.dtype)
            else:
                query_states, key_states, value_states, conv_states[0], conv_states[1] = P.swa_qkv_conv_clip(
                    q_raw,
                    k_raw,
                    v_raw,
                    self.q_l2_clip.scale,
                    self.k_sconv.conv1d.weight,
                    self.v_sconv.conv1d.weight,
                    conv_states[0],
                    conv_states[1],
                    num_heads=self.num_heads,
                    num_kv_heads=self.num_key_value_heads,
                    thr_q=self.q_l2_clip.threshold,
                    thr_k=self.k_l2_clip.threshold,
                    eps=self.eps,
                    conv_mask=conv_mask,
                    conv_lengths=conv_lengths,
                )
                relative_states = self.rel_l2_clip(r_raw.reshape(B, L, self.num_heads, -1))
                max_dist = L if kv_cache is None else (kv_cache.offset + L if isinstance(kv_cache.offset, int) else None)
                md = ((max_dist + 63) // 64) * 64 if max_dist is not None else None
                if md is not None and 0 < md < self.rel_logits_proj.proj.shape[1]:
                    p = self._rel_proj_bf16(relative_states.dtype)
                    position_bias = relative_states @ p[:, :md]
                else:
                    position_bias = self.rel_logits_proj(relative_states)
                rel_proj = None
            if kv_cache is not None:
                key_states, value_states = kv_cache.update_and_fetch(key_states, value_states)
            attn_output = self._attention(
                query_states, key_states, value_states, position_bias, sinks, kv_cache, rel_proj=rel_proj
            )
            return self._attend_post(attn_output, gate)
        key_states, conv_states[0] = self.k_sconv(
            P.aff8_linear(self.k_proj, key_hidden_states) if use_a8 else self.k_proj(key_hidden_states),
            conv_states[0],
            conv_mask,
            fast=fast_conv,
            conv_lengths=conv_lengths,
        )
        value_states, conv_states[1] = self.v_sconv(
            P.aff8_linear(self.v_proj, value_hidden_states) if use_a8 else self.v_proj(value_hidden_states),
            conv_states[1],
            conv_mask,
            fast=fast_conv,
            conv_lengths=conv_lengths,
        )
        return self._attend(
            hidden_states,
            P.aff8_linear(self.q_proj, hidden_states) if use_a8 else self.q_proj(hidden_states),
            key_states,
            value_states,
            P.aff8_linear(self.r_proj, hidden_states) if use_a8 else self.r_proj(hidden_states),
            kv_cache,
        )


class FullAttention(Attention):
    def __init__(self, args: TextArgs):
        super().__init__(args, sliding=False)
        dims = self.num_key_value_heads * self.head_dim
        conv_args = (args.hidden_size, self.num_key_value_heads, args.sconv_kernel_size)
        self.k_sconv = DynamicShortConvolution(dims, *conv_args)
        self.v_sconv = DynamicShortConvolution(dims, *conv_args)
        self.log_scaling_n_floor = args.log_scaling_n_floor
        self.log_scaling_alpha = args.log_scaling_alpha

    def __call__(
        self,
        hidden_states: mx.array,
        key_hidden_states: mx.array,
        value_hidden_states: mx.array,
        *,
        cache: Any | None,
        conv_states: list,
        conv_mask: mx.array | None,
        fast_conv: bool = False,
        conv_lengths: mx.array | None = None,
    ) -> mx.array:
        kv_cache = cache[0] if cache is not None else None
        use_a8 = P._PF_AFF8_QMM and P.enabled(hidden_states)
        key_states, conv_states[0] = self.k_sconv(
            P.aff8_linear(self.k_proj, key_hidden_states) if use_a8 else self.k_proj(key_hidden_states),
            key_hidden_states,
            conv_states[0],
            conv_mask,
            fast=fast_conv,
            conv_lengths=conv_lengths,
        )
        value_states, conv_states[1] = self.v_sconv(
            P.aff8_linear(self.v_proj, value_hidden_states) if use_a8 else self.v_proj(value_hidden_states),
            value_hidden_states,
            conv_states[1],
            conv_mask,
            fast=fast_conv,
            conv_lengths=conv_lengths,
        )

        offset = kv_cache.offset if kv_cache is not None else 0
        position_ids = _positions(offset, hidden_states.shape[1]).astype(mx.float32)
        tau = 1.0 + self.log_scaling_alpha * mx.log(
            mx.maximum((position_ids + 1) / self.log_scaling_n_floor, 1.0)
        )
        return self._attend(
            hidden_states,
            P.aff8_linear(self.q_proj, hidden_states) if use_a8 else self.q_proj(hidden_states),
            key_states,
            value_states,
            P.aff8_linear(self.r_proj, hidden_states) if use_a8 else self.r_proj(hidden_states),
            kv_cache,
            tau=tau[..., None, None],
        )


class SoftcappedSwiGLU(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.gate_softcap = args.mlp_gate_softcap
        self.up_softcap = args.mlp_up_softcap

    def __call__(self, gate: mx.array, up: mx.array) -> mx.array:
        return _softcapped_swiglu(gate, up, self.gate_softcap, self.up_softcap)


class Experts(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        dims, hidden = args.hidden_size, args.intermediate_size
        num_experts = args.n_routed_experts
        # The rows alternate between gate and up, as in the checkpoint.
        self.gate_up_proj = SwitchLinear(dims, 2 * hidden, num_experts, bias=False)
        self.down_proj = SwitchLinear(hidden, dims, num_experts, bias=False)
        self.act_fn = SoftcappedSwiGLU(args)

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))
        indices = mx.stop_gradient(indices)
        do_sort = indices.size >= 64
        idx, inv_order = indices, None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        gate_up = self.gate_up_proj(x, idx, sorted_indices=do_sort)
        x = self.act_fn(gate_up[..., 0::2], gate_up[..., 1::2])
        x = self.down_proj(x, idx, sorted_indices=do_sort)
        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)
        return x.squeeze(-2)


class TopKRouter(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.routed_scaling_factor = args.route_scale
        num_experts, dims = args.n_routed_experts, args.hidden_size
        self.weight = mx.zeros((num_experts, dims), dtype=mx.float32)
        self.e_score_correction_bias = mx.zeros((num_experts,), dtype=mx.float32)

    def __call__(self, hidden_states: mx.array) -> tuple[mx.array, mx.array]:
        if (
            P._PF_ROUTER
            and P.enabled(hidden_states)
            and hidden_states.dtype == mx.bfloat16
            and self.weight.dtype == mx.float32
            and hidden_states.shape[-1] == P._D
            and self.weight.shape == (128, P._D)
            and hidden_states.shape[0] * hidden_states.shape[1] >= 4096
        ):
            logits = P.router_logits_bf16(hidden_states, self.weight)
        else:
            logits = hidden_states.astype(mx.float32) @ self.weight.T
        if _use_kernels():
            return expert_routing(
                logits,
                self.e_score_correction_bias,
                top_k=self.top_k,
                multiplier=1.0,
                score_eps=_ROUTER_SCORE_EPS,
                route_scale=self.routed_scaling_factor,
            )
        scores = mx.sigmoid(logits)
        biased = scores + self.e_score_correction_bias
        k = self.top_k
        indices = mx.argpartition(-biased, kth=k - 1, axis=-1)[..., :k]
        weights = mx.take_along_axis(scores, indices, axis=-1)
        weights = weights / (weights.sum(axis=-1, keepdims=True) + _ROUTER_SCORE_EPS)
        return indices, weights * self.routed_scaling_factor


class SparseMoeBlock(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.gate = TopKRouter(args)
        self.experts = Experts(args)
        self.sharding_group = None

    def __call__(self, hidden_states: mx.array) -> mx.array:
        group = self.sharding_group
        if group is not None:
            hidden_states = sum_gradients(group)(hidden_states)
        selected_experts, routing_weights = self.gate(hidden_states)
        if (
            group is None
            and P.enabled(hidden_states)
            and selected_experts.size >= 16
            and selected_experts.shape[-1] == 8
        ):
            ex = self.experts
            idx = mx.stop_gradient(selected_experts)
            if P.moe_gmm_ok(ex, hidden_states, idx):
                return P.moe_gmm_forward(ex, hidden_states, idx, routing_weights)
            if P.moe_short_ok(ex, hidden_states, idx):
                return P.moe_short_forward(ex, hidden_states, idx, routing_weights)
            num_e = ex.gate_up_proj.num_experts
            if num_e % 32 == 0 and num_e <= 256 and hidden_states.shape[-1] % 8 == 0:
                x, idx_s, inv_order = P.moe_gather_sort(hidden_states, idx, num_experts=num_e)
            else:
                x = mx.expand_dims(hidden_states, (-2, -3))
                x, idx_s, inv_order = _gather_sort(x, idx)
            gu = ex.gate_up_proj(x, idx_s, sorted_indices=True)
            x = ex.act_fn(gu[..., 0::2], gu[..., 1::2])
            dn_s = ex.down_proj(x, idx_s, sorted_indices=True)
            return P.unsort_combine(dn_s, inv_order, routing_weights)
        expert_states = self.experts(hidden_states, selected_experts)
        expert_states = expert_states.astype(mx.float32) * routing_weights[..., None]
        final_hidden_states = expert_states.sum(axis=-2)
        if group is not None:
            final_hidden_states = mx.distributed.all_sum(
                final_hidden_states, group=group
            )
        return final_hidden_states.astype(hidden_states.dtype)


class BlockAttentionResidual(nn.Module):
    def __init__(self, dims: int, num_queries: int = 1):
        super().__init__()
        shape = (dims,) if num_queries == 1 else (num_queries, dims)
        self.query = mx.zeros(shape, dtype=mx.float32)

    def __call__(
        self, block_states: mx.array, partial_block_states: mx.array
    ) -> mx.array:
        dims = partial_block_states.shape[-1]
        return _depth_attention(
            block_states, partial_block_states, self.query.reshape(-1, dims)
        )


def _depth_attention(
    block_states: mx.array, partial_block_states: mx.array, query: mx.array
) -> mx.array:
    query = query.astype(mx.float32)
    scale = partial_block_states.shape[-1] ** -0.5
    if _use_kernels():
        return depth_attention(block_states, partial_block_states, query, scale=scale)
    stacked_states = mx.concatenate([block_states, partial_block_states[None]])
    stacked_states = mx.moveaxis(stacked_states, 0, -2)
    weights = mx.softmax((stacked_states @ query.T) * scale, axis=-2, precise=True)
    return weights.swapaxes(-1, -2) @ stacked_states


class MessageGate(nn.Module):
    def __init__(self, dims: int, rank: int):
        super().__init__()
        self.down_proj = nn.Linear(2 * dims, rank, bias=False)
        self.forget_gate_proj = nn.Linear(rank, dims, bias=False)
        self.message_gate_proj = nn.Linear(rank, dims, bias=False)

    def __call__(self, hidden_states: mx.array, message: mx.array) -> mx.array:
        states = mx.concatenate(
            [unweighted_rms_norm(hidden_states), unweighted_rms_norm(message)],
            axis=-1,
        )
        states = self.down_proj(states.astype(self.down_proj.weight.dtype))
        states = states.astype(mx.float32)
        forget_gate = mx.tanh(self.forget_gate_proj(states))
        message_gate = mx.tanh(self.message_gate_proj(states))
        return (1 - forget_gate) * hidden_states + message_gate * message


class MessageInput(nn.Module):
    def __init__(self, dims: int, num_looped_layers: int):
        super().__init__()
        # The query, key, value and mlp queries of each looped layer.
        self.query = mx.zeros((num_looped_layers, 4, dims), mx.float32)

    def __call__(self, bank: tuple, loop_layer_idx: int) -> mx.array:
        return _depth_attention(*bank, self.query[loop_layer_idx])


class DecoderLayer(nn.Module):
    def __init__(self, args: TextArgs, layer_idx: int, *, num_queries: int):
        super().__init__()
        dims = args.hidden_size
        self.is_block_boundary = layer_idx % args.n_layers_per_attn_res_block == 0
        self.attention_input = (
            BlockAttentionResidual(dims, num_queries) if layer_idx > 0 else None
        )
        self.mlp_input = BlockAttentionResidual(dims)
        self.mlp = SparseMoeBlock(args)
        self.post_attention_layernorm = nn.RMSNorm(dims, eps=args.rms_norm_eps)
        kernel_size = args.sconv_kernel_size
        self.post_attention_convolution = ShortConvolution(dims, kernel_size)
        self.post_mlp_convolution = ShortConvolution(dims, kernel_size)
        if layer_idx in args.loop_core:
            rank = args.loop_config["gate_rank"]
            self.q_message_gate = MessageGate(dims, rank)
            self.k_message_gate = MessageGate(dims, rank)
            self.v_message_gate = MessageGate(dims, rank)
            self.mlp_message_gate = MessageGate(dims, rank)

    def _mix_messages(self, states: tuple, messages: mx.array) -> tuple:
        q_states, k_states, v_states = states
        return (
            self.q_message_gate(q_states, messages[..., 0, :]),
            self.k_message_gate(k_states, messages[..., 1, :]),
            self.v_message_gate(v_states, messages[..., 2, :]),
        )

    def __call__(
        self,
        hidden_states: mx.array,
        block_states: mx.array | None,
        *,
        cache: Any | None = None,
        conv_mask: mx.array | None = None,
        messages: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        if _FAST_ON and fastdecode.eligible(hidden_states, cache, conv_mask):
            return fastdecode.decode_layer(self, hidden_states, block_states, cache, messages)
        if isinstance(block_states, list):
            block_states = None if not block_states else mx.stack(block_states, axis=0)[:, :, None, :]
        if messages is not None and messages.ndim == 3:
            messages = messages[:, None, :, :]
        dtype = hidden_states.dtype
        conv = cache[1] if cache is not None else None
        conv_states = list(conv.cache) if conv is not None else [None] * 4
        conv_lengths = conv.lengths if conv is not None else None
        fast_conv = (
            P.enabled(hidden_states)
            and (conv is None or not getattr(conv, "record", False))
        )
        block_residual = hidden_states
        dtype = hidden_states.dtype
        use_fused_attn_in = (
            P._PF_DEPTH_RMS
            and P.active(dtype, hidden_states)
            and messages is None
            and isinstance(self, SlidingDecoderLayer)
            and self.attention_input is not None
            and block_states is not None
            and hidden_states.shape[-1] == P._D
            and self.input_layernorm.weight.dtype == mx.bfloat16
            and self.attention_input.query.dtype == mx.float32
        )
        if use_fused_attn_in:
            attention_inputs, normed_residual = P.unw_rms_depth_attn_rms_bf16(
                block_residual, block_states, self.attention_input.query,
                self.input_layernorm, need_nr=self.is_block_boundary,
            )
        elif (
            P._PF_DEPTH_RMS
            and P.active(dtype, hidden_states)
            and self.attention_input is not None
            and block_states is not None
            and hidden_states.shape[-1] == P._D
            and self.attention_input.query.dtype == mx.float32
        ):
            attention_inputs, normed_residual = P.unw_rms_depth_attn_f32(
                block_residual, block_states, self.attention_input.query,
                need_nr=self.is_block_boundary,
            )
        else:
            normed_residual = unweighted_rms_norm(block_residual.astype(mx.float32))
            attention_inputs = None
            if self.attention_input is not None:
                attention_inputs = self.attention_input(block_states, normed_residual)
        hidden_states = self._self_attention(
            hidden_states, attention_inputs, cache, conv_states, conv_mask, messages,
            fast_conv=fast_conv, conv_lengths=conv_lengths,
        )
        can_sc_post = (
            fast_conv
            and self.post_attention_convolution.kernel_size == 4
            and hidden_states.shape[-1] == P._D
            and self.post_attention_convolution.conv1d.weight.dtype == mx.bfloat16
            and (conv_states[2] is None or conv_states[2].dtype == mx.float32)
        )
        if can_sc_post and P._PF_DEPTH_RMS and P.active(dtype, block_residual) and self.mlp_input.query.dtype == mx.float32:
            if self.is_block_boundary:
                block = normed_residual[None]
                block_states = block if block_states is None else mx.concatenate([block_states, block])
            if block_states is not None and (
                messages is not None or self.post_attention_layernorm.weight.dtype == mx.bfloat16
            ):
                ln_post = self.post_attention_layernorm if messages is None else None
                residual, da_out, conv_states[2] = P.short_conv_attn_depth_attn(
                    hidden_states,
                    conv_states[2],
                    self.post_attention_convolution.conv1d.weight,
                    None if self.is_block_boundary else block_residual,
                    block_states,
                    self.mlp_input.query,
                    ln=ln_post,
                    conv_mask=conv_mask,
                    conv_lengths=conv_lengths,
                )
                if messages is None:
                    mlp_in = da_out
                elif P.msg_gate_ok(self.mlp_message_gate, da_out, messages, self.post_attention_layernorm):
                    mlp_in = P.msg_gate_norm_single(
                        self.mlp_message_gate, da_out, messages, 3, self.post_attention_layernorm
                    )
                else:
                    mlp_in = self.post_attention_layernorm(
                        self.mlp_message_gate(da_out, messages[..., 3, :])
                    ).astype(dtype)
                normed_states = None
            else:
                residual, normed_states, conv_states[2] = P.short_conv_attn_post(
                    hidden_states,
                    conv_states[2],
                    self.post_attention_convolution.conv1d.weight,
                    None if self.is_block_boundary else block_residual,
                    conv_mask=conv_mask,
                    conv_lengths=conv_lengths,
                )
        elif can_sc_post:
            residual, normed_states, conv_states[2] = P.short_conv_attn_post(
                hidden_states,
                conv_states[2],
                self.post_attention_convolution.conv1d.weight,
                None if self.is_block_boundary else block_residual,
                conv_mask=conv_mask,
                conv_lengths=conv_lengths,
            )
            if self.is_block_boundary:
                block = normed_residual[None]
                if block_states is None:
                    block_states = block
                else:
                    block_states = mx.concatenate([block_states, block])
        else:
            hidden_states, conv_states[2] = self.post_attention_convolution(
                hidden_states, conv_states[2], conv_mask, fast=fast_conv, conv_lengths=conv_lengths
            )

            residual = hidden_states = hidden_states.astype(mx.float32)
            if self.is_block_boundary:
                block = normed_residual[None]
                if block_states is None:
                    block_states = block
                else:
                    block_states = mx.concatenate([block_states, block])
            else:
                residual = hidden_states = block_residual.astype(mx.float32) + hidden_states

            normed_states = unweighted_rms_norm(hidden_states)
        if normed_states is not None:
            if (
                messages is None
                and P._PF_DEPTH_RMS
                and P.active(dtype, block_residual)
                and block_states is not None
                and normed_states.dtype == mx.float32
                and normed_states.shape[-1] == P._D
                and self.post_attention_layernorm.weight.dtype == mx.bfloat16
                and self.mlp_input.query.dtype == mx.float32
            ):
                mlp_in = P.depth_attn_rms_bf16(
                    block_states, normed_states, self.mlp_input.query, self.post_attention_layernorm
                )
            else:
                hidden_states = self.mlp_input(block_states, normed_states)[..., 0, :]
                if (
                    messages is not None
                    and P.active(dtype, block_residual)
                    and P.msg_gate_ok(self.mlp_message_gate, hidden_states, messages, self.post_attention_layernorm)
                ):
                    mlp_in = P.msg_gate_norm_single(
                        self.mlp_message_gate, hidden_states, messages, 3, self.post_attention_layernorm
                    )
                elif (
                    messages is None
                    and P.active(dtype, block_residual)
                    and hidden_states.dtype == mx.float32
                    and hidden_states.shape[-1] == P._D
                    and self.post_attention_layernorm.weight.dtype == mx.bfloat16
                ):
                    mlp_in = P.rms_norm_f32_bf16(hidden_states, self.post_attention_layernorm)
                else:
                    if messages is not None:
                        hidden_states = self.mlp_message_gate(hidden_states, messages[..., 3, :])
                    mlp_in = self.post_attention_layernorm(hidden_states).astype(dtype)
        hidden_states = self.mlp(mlp_in)
        fused_mlp_post = (
            fast_conv
            and self.post_mlp_convolution.kernel_size == 4
            and hidden_states.shape[1] >= 2
            and hidden_states.shape[-1] % 4 == 0
            and residual.dtype == mx.float32
            and self.post_mlp_convolution.conv1d.weight.dtype == mx.bfloat16
            and (conv_states[3] is None or conv_states[3].dtype == mx.float32)
        )
        if fused_mlp_post:
            hidden_states, conv_states[3] = P.short_conv_residual(
                hidden_states,
                conv_states[3],
                self.post_mlp_convolution.conv1d.weight,
                residual,
                conv_mask=conv_mask,
                conv_lengths=conv_lengths,
            )
        else:
            hidden_states, conv_states[3] = self.post_mlp_convolution(
                hidden_states, conv_states[3], conv_mask, fast=fast_conv, conv_lengths=conv_lengths
            )

        if conv is not None:
            if getattr(conv, "record", False):  # LOOPLING_MTP: keep full windows for rollback
                conv.windows = list(conv_states)
            n = self.post_mlp_convolution.kernel_size - 1
            if getattr(conv, "record", False) and block_residual.shape[1] > 1:
                conv.windows = list(conv_states)  # LOOPLING_SELFSPEC: windows to roll back to
            for i, window in enumerate(conv_states):
                conv[i] = window if window.shape[1] == n else _conv_state(window, conv.lengths, n)
            conv.advance(block_residual.shape[1])
        if fused_mlp_post:
            return hidden_states, block_states
        hidden_states = residual + hidden_states.astype(mx.float32)
        return hidden_states.astype(dtype), block_states


class SlidingDecoderLayer(DecoderLayer):
    def __init__(self, args: TextArgs, layer_idx: int):
        super().__init__(args, layer_idx, num_queries=1)
        self.self_attn = SlidingAttention(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def _self_attention(
        self, hidden_states, attention_inputs, cache, conv_states, conv_mask, messages, *, fast_conv=False, conv_lengths=None
    ):
        dtype = hidden_states.dtype
        if attention_inputs is not None and attention_inputs.dtype == dtype and attention_inputs.ndim == 3:
            states = (attention_inputs,) * 3
        else:
            if attention_inputs is not None:
                hidden_states = attention_inputs[..., 0, :]
            if messages is None:
                if (
                    P.active(dtype, hidden_states)
                    and hidden_states.dtype == mx.float32
                    and hidden_states.shape[-1] == P._D
                    and self.input_layernorm.weight.dtype == mx.bfloat16
                ):
                    states = (P.rms_norm_f32_bf16(hidden_states, self.input_layernorm),) * 3
                else:
                    states = (self.input_layernorm(hidden_states).astype(dtype),) * 3
            elif (
                P.active(dtype, hidden_states)
                and P.msg_gate_ok(self.q_message_gate, hidden_states, messages, self.input_layernorm)
                and P.msg_gate_ok(self.k_message_gate, hidden_states, messages, self.input_layernorm)
                and P.msg_gate_ok(self.v_message_gate, hidden_states, messages, self.input_layernorm)
            ):
                states = P.msg_gate_norm_qkv(
                    (self.q_message_gate, self.k_message_gate, self.v_message_gate),
                    hidden_states,
                    messages,
                    (self.input_layernorm,) * 3,
                    shared_h=True,
                    shared_ln=True,
                )
            else:
                states = self._mix_messages((hidden_states,) * 3, messages)
                states = [self.input_layernorm(s).astype(dtype) for s in states]
        return self.self_attn(
            *states,
            cache=cache,
            conv_states=conv_states,
            conv_mask=conv_mask,
            fast_conv=fast_conv,
            conv_lengths=conv_lengths,
        )


class FullDecoderLayer(DecoderLayer):
    def __init__(self, args: TextArgs, layer_idx: int):
        super().__init__(args, layer_idx, num_queries=3)
        self.self_attn = FullAttention(args)
        eps = args.rms_norm_eps
        self.q_input_layernorm = nn.RMSNorm(args.hidden_size, eps=eps)
        self.k_input_layernorm = nn.RMSNorm(args.hidden_size, eps=eps)
        self.v_input_layernorm = nn.RMSNorm(args.hidden_size, eps=eps)

    def _self_attention(
        self, hidden_states, attention_inputs, cache, conv_states, conv_mask, messages, *, fast_conv=False, conv_lengths=None
    ):
        dtype = hidden_states.dtype
        lns = (self.q_input_layernorm, self.k_input_layernorm, self.v_input_layernorm)
        if (
            messages is not None
            and attention_inputs is not None
            and P.active(dtype, hidden_states)
            and lns[0].eps == lns[1].eps == lns[2].eps
            and P.msg_gate_ok(self.q_message_gate, attention_inputs, messages, lns[0])
            and P.msg_gate_ok(self.k_message_gate, attention_inputs, messages, lns[1])
            and P.msg_gate_ok(self.v_message_gate, attention_inputs, messages, lns[2])
        ):
            states = P.msg_gate_norm_qkv(
                (self.q_message_gate, self.k_message_gate, self.v_message_gate),
                attention_inputs,
                messages,
                lns,
                shared_h=False,
                shared_ln=False,
            )
            return self.self_attn(
                *states,
                cache=cache,
                conv_states=conv_states,
                conv_mask=conv_mask,
                fast_conv=fast_conv,
                conv_lengths=conv_lengths,
            )
        if (
            messages is None
            and attention_inputs is not None
            and P.active(dtype, hidden_states)
            and attention_inputs.dtype == mx.float32
            and attention_inputs.shape[-1] == P._D
            and lns[0].eps == lns[1].eps == lns[2].eps
            and lns[0].weight.dtype == lns[1].weight.dtype == lns[2].weight.dtype == mx.bfloat16
        ):
            states = P.rms_norm_f32_bf16_qkv(attention_inputs, lns)
            return self.self_attn(
                *states,
                cache=cache,
                conv_states=conv_states,
                conv_mask=conv_mask,
                fast_conv=fast_conv,
                conv_lengths=conv_lengths,
            )
        if attention_inputs is None:
            query_states = key_states = value_states = hidden_states
        else:
            query_states, key_states, value_states = (
                attention_inputs[..., i, :] for i in range(3)
            )
        if messages is not None:
            query_states, key_states, value_states = self._mix_messages(
                (query_states, key_states, value_states), messages
            )
        return self.self_attn(
            self.q_input_layernorm(query_states).astype(dtype),
            self.k_input_layernorm(key_states).astype(dtype),
            self.v_input_layernorm(value_states).astype(dtype),
            cache=cache,
            conv_states=conv_states,
            conv_mask=conv_mask,
            fast_conv=fast_conv,
            conv_lengths=conv_lengths,
        )
