"""DFlash 2 block-diffusion speculative drafter for Inkling2 / Loopling (opt-in LOOPLING_MTP=dflash2).

Ports Splash's `incoai/Inkling-2-Mini-DFlash2` model (5-layer non-causal Qwen3 backbone with 2-tap dynamic convolutions
and low-rank bilinear candidate selector) paired with BatchDecoder's bit-exact causal verify engine.
"""
from __future__ import annotations

import heapq
import json
import logging
import math
import os
import time
from collections import deque

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_lm.models.inkling2 import fastdecode as FD
from mlx_lm.models.inkling2.rows import qmv as RQ
from mlx_lm.models.inkling2.rows import qmv4 as RQ4

_USE_QMV4 = os.environ.get("LOOPLING_DFLASH2_QMV4", "0") not in ("0", "", "false", "False")
_LOOKUP_BYPASS_MIN = int(os.environ.get("LOOPLING_DFLASH2_LOOKUP_BYPASS", "6"))
_LOOKUP_W = max(8, min(16, int(os.environ.get("LOOPLING_DFLASH2_LOOKUP_W", "16"))))

DFLASH2_DIR = os.environ.get(
    "LOOPLING_DFLASH2_DIR",
    os.path.expanduser("~/loopling/models/Inkling-2-Mini-DFlash2"),
)
_EPS = 1e-6


def _quantize_if(w: mx.array, bits: int, *, fp16: bool = False):
    if fp16:
        w = w.astype(mx.float16)
    if bits == 4:
        return mx.quantize(w, group_size=64, bits=4, mode="affine")
    if bits == 8:
        return mx.quantize(w, group_size=64, bits=8, mode="affine")
    return mx.contiguous(w.T)


def _rms(x: mx.array, w: mx.array) -> mx.array:
    return mx.fast.rms_norm(x, w, _EPS)


@mx.compile
def _dyn_conv(x: mx.array, dyn: mx.array, base: mx.array, residual: mx.array | None = None) -> mx.array:
    """2-tap per-group dynamic convolution (Splash draft.metal::draft_conv_phase)."""
    n_rows, hidden = x.shape
    stage = 0 if residual is None else 1
    d0 = mx.broadcast_to(dyn[:, stage, 0, :, None], (n_rows, 144, 16)).reshape(n_rows, hidden)
    d1 = mx.broadcast_to(dyn[:, stage, 1, :, None], (n_rows, 144, 16)).reshape(n_rows, hidden)
    w0 = base[stage, 0].astype(mx.float32) + d0.astype(mx.float32)
    w1 = base[stage, 1].astype(mx.float32) + d1.astype(mx.float32)
    xf = x.astype(mx.float32)
    prev = mx.concatenate([mx.zeros((1, hidden), mx.float32), xf[:-1]], axis=0)
    val = xf * w0 + prev * w1
    if residual is not None:
        val = val + residual.astype(mx.float32)
    return val.astype(mx.bfloat16)


@mx.compile
def _dyn_conv_batch(x: mx.array, dyn: mx.array, base: mx.array, residual: mx.array | None = None) -> mx.array:
    """Batched 2-tap per-group dynamic conv over `(C, 8, H)` without cross-stream bleed."""
    C_n, _, hidden = x.shape
    stage = 0 if residual is None else 1
    d0 = mx.broadcast_to(dyn[:, :, stage, 0, :, None], (C_n, 8, 144, 16)).reshape(C_n, 8, hidden)
    d1 = mx.broadcast_to(dyn[:, :, stage, 1, :, None], (C_n, 8, 144, 16)).reshape(C_n, 8, hidden)
    w0 = base[stage, 0].astype(mx.float32) + d0.astype(mx.float32)
    w1 = base[stage, 1].astype(mx.float32) + d1.astype(mx.float32)
    xf = x.astype(mx.float32)
    prev = mx.concatenate([mx.zeros((C_n, 1, hidden), mx.float32), xf[:, :-1, :]], axis=1)
    val = xf * w0 + prev * w1
    if residual is not None:
        val = val + residual.astype(mx.float32)
    return val.astype(mx.bfloat16)


@mx.compile
def _swiglu(g: mx.array, u: mx.array) -> mx.array:
    return nn.silu(g) * u


@mx.compile
def _qkv_norm_split(
    qkv: mx.array,
    q_norm_w: mx.array,
    k_norm_w: mx.array,
    nh: int,
    nkv: int,
    hd: int,
):
    B = qkv.shape[0]
    q_w = nh * hd
    k_w = nkv * hd
    q = mx.fast.rms_norm(qkv[:, :q_w].reshape(1, B, nh, hd), q_norm_w, _EPS).transpose(0, 2, 1, 3)
    k = mx.fast.rms_norm(qkv[:, q_w : q_w + k_w].reshape(1, B, nkv, hd), k_norm_w, _EPS).transpose(0, 2, 1, 3)
    v = qkv[:, q_w + k_w :].reshape(1, B, nkv, hd).transpose(0, 2, 1, 3)
    return q, k, v


_DF_COMPILED: dict = {}


def _get_df_compiled(nh: int, nkv: int, hd: int, inter: int, qbits: int):
    key = (nh, nkv, hd, inter, qbits)
    ent = _DF_COMPILED.get(key)
    if ent is not None:
        return ent

    def _dlin(x: mx.array, w) -> mx.array:
        if qbits:
            if (
                qbits == 4
                and _USE_QMV4
                and x.ndim == 2
                and x.dtype == mx.bfloat16
                and x.shape[0] <= 16
                and x.shape[1] % 256 == 0
                and x.shape[1] not in (64, 128)
                and w[1].dtype == mx.bfloat16
            ):
                return RQ4.qmv4_rows(x, w[0], w[1], w[2])
            return mx.quantized_matmul(
                x, w[0], scales=w[1], biases=w[2], transpose=True, group_size=64, bits=qbits, mode="affine"
            )
        return x @ w

    @mx.compile
    def pre_rope(x, in_ln, a_dyn, a_base, qkv_w, q_norm, k_norm):
        B = x.shape[0]
        h_norm = mx.fast.rms_norm(x, in_ln, _EPS)
        dyn_a = _dlin(h_norm, a_dyn).reshape(B, 2, 2, 144)
        c_in = _dyn_conv(h_norm, dyn_a, a_base, None)
        qkv = _dlin(c_in, qkv_w)
        q_w = nh * hd
        k_w = nkv * hd
        q = mx.fast.rms_norm(qkv[:, :q_w].reshape(1, B, nh, hd), q_norm, _EPS).transpose(0, 2, 1, 3)
        k = mx.fast.rms_norm(qkv[:, q_w : q_w + k_w].reshape(1, B, nkv, hd), k_norm, _EPS).transpose(0, 2, 1, 3)
        v = qkv[:, q_w + k_w :].reshape(1, B, nkv, hd).transpose(0, 2, 1, 3)
        return q, k, v, dyn_a

    @mx.compile
    def post_attn(x, attn, o_w, dyn_a, a_base, post_ln, m_dyn, m_base, gu_w, dn_w):
        B = x.shape[0]
        proj = _dlin(attn, o_w)
        res = _dyn_conv(proj, dyn_a, a_base, x)
        m_norm = mx.fast.rms_norm(res, post_ln, _EPS)
        dyn_m = _dlin(m_norm, m_dyn).reshape(B, 2, 2, 144)
        m_in = _dyn_conv(m_norm, dyn_m, m_base, None)
        gu = _dlin(m_in, gu_w)
        m_out = _dlin(_swiglu(gu[:, :inter], gu[:, inter:]), dn_w)
        return _dyn_conv(m_out, dyn_m, m_base, res)

    ent = _DF_COMPILED[key] = (pre_rope, post_attn)
    return ent


_DF_BATCH_COMPILED: dict = {}


def _get_df_batch_compiled(nh: int, nkv: int, hd: int, inter: int, qbits: int, theta: float):
    key = (nh, nkv, hd, inter, qbits, float(theta))
    ent = _DF_BATCH_COMPILED.get(key)
    if ent is not None:
        return ent

    def _dlin(x: mx.array, w) -> mx.array:
        if qbits:
            if (
                qbits == 4
                and _USE_QMV4
                and x.ndim == 2
                and x.dtype == mx.bfloat16
                and x.shape[0] <= 16
                and x.shape[1] % 256 == 0
                and x.shape[1] not in (64, 128)
                and w[1].dtype == mx.bfloat16
            ):
                return RQ4.qmv4_rows(x, w[0], w[1], w[2])
            return mx.quantized_matmul(
                x, w[0], scales=w[1], biases=w[2], transpose=True, group_size=64, bits=qbits, mode="affine"
            )
        return x @ w

    @mx.compile
    def pre_rope_b(x_3d, in_ln, a_dyn, a_base, qkv_w, q_norm, k_norm, offs_arr):
        Cn, blk, H = x_3d.shape
        B_tot = Cn * blk
        h_norm = mx.fast.rms_norm(x_3d.reshape(B_tot, H), in_ln, _EPS).reshape(Cn, blk, H)
        dyn_a = _dlin(h_norm.reshape(B_tot, H), a_dyn).reshape(Cn, blk, 2, 2, 144)
        c_in = _dyn_conv_batch(h_norm, dyn_a, a_base, None)
        qkv = _dlin(c_in.reshape(B_tot, H), qkv_w)
        q_w = nh * hd
        k_w = nkv * hd
        q_raw = mx.fast.rms_norm(qkv[:, :q_w].reshape(Cn, blk, nh, hd), q_norm, _EPS).transpose(0, 2, 1, 3)
        k_raw = mx.fast.rms_norm(qkv[:, q_w : q_w + k_w].reshape(Cn, blk, nkv, hd), k_norm, _EPS).transpose(0, 2, 1, 3)
        v_cur = qkv[:, q_w + k_w :].reshape(Cn, blk, nkv, hd).transpose(0, 2, 1, 3)
        q_cur = mx.fast.rope(q_raw, hd, traditional=False, base=theta, scale=1.0, offset=offs_arr)
        k_cur = mx.fast.rope(k_raw, hd, traditional=False, base=theta, scale=1.0, offset=offs_arr)
        return q_cur, k_cur, v_cur, dyn_a

    @mx.compile
    def post_attn_b(x_3d, attn_2d, o_w, dyn_a, a_base, post_ln, m_dyn, m_base, gu_w, dn_w):
        Cn, blk, H = x_3d.shape
        B_tot = Cn * blk
        proj = _dlin(attn_2d, o_w).reshape(Cn, blk, H)
        res = _dyn_conv_batch(proj, dyn_a, a_base, x_3d)
        m_norm = mx.fast.rms_norm(res.reshape(B_tot, H), post_ln, _EPS).reshape(Cn, blk, H)
        dyn_m = _dlin(m_norm.reshape(B_tot, H), m_dyn).reshape(Cn, blk, 2, 2, 144)
        m_in = _dyn_conv_batch(m_norm, dyn_m, m_base, None)
        gu = _dlin(m_in.reshape(B_tot, H), gu_w)
        m_out = _dlin(_swiglu(gu[:, :inter], gu[:, inter:]), dn_w).reshape(Cn, blk, H)
        return _dyn_conv_batch(m_out, dyn_m, m_base, res)

    @mx.compile
    def sel_n1(raw_logits, sel_h, pred_cb, succ_cb, anc_nn, softcap: float):
        top_idx = mx.argpartition(-raw_logits, 16, axis=-1)[:, :16]
        unary = mx.take_along_axis(raw_logits, top_idx, axis=-1).astype(mx.float32)
        succ0 = succ_cb[top_idx].astype(mx.float32)
        pred0 = pred_cb[anc_nn].astype(mx.float32) * sel_h
        edge0 = mx.matmul(succ0, pred0[:, :, None])[:, :, 0]
        sum0 = unary + edge0
        sc0 = softcap * mx.tanh(sum0 / softcap) if softcap > 0.0 else sum0
        best_slot = mx.argmax(sc0, axis=-1)
        return mx.take_along_axis(top_idx, best_slot[:, None], axis=-1)[:, 0]

    ent = _DF_BATCH_COMPILED[key] = (pre_rope_b, post_attn_b, sel_n1)
    return ent


@mx.compile
def _selector_scores_from_top16(
    top_idx: mx.array,
    unary: mx.array,
    sel_h: mx.array,
    pred_cb: mx.array,
    succ_cb: mx.array,
    anchor_tok: mx.array,
    softcap: float,
):
    """Bigram codebook scores from pre-extracted Top-16 `(top_idx (P, 16), unary (P, 16) f32)`."""
    succ_vecs = succ_cb[top_idx].astype(mx.float32)  # (P, 16, R)

    anc = anchor_tok.reshape(())
    pred0 = pred_cb[anc].astype(mx.float32) * sel_h[0]  # (R,)
    edge0 = succ_vecs[0] @ pred0  # (16,)
    sum0 = unary[0] + edge0
    sc0 = softcap * mx.tanh(sum0 / softcap) if softcap > 0.0 else sum0

    pred_vecs = pred_cb[top_idx[:-1]].astype(mx.float32) * sel_h[1:, None, :]  # (P-1, 16, R)
    edges = mx.matmul(pred_vecs, succ_vecs[1:].swapaxes(-1, -2))  # (P-1, 16, 16)
    sums = unary[1:, None, :] + edges
    sc16 = softcap * mx.tanh(sums / softcap) if softcap > 0.0 else sums
    return top_idx, sc0, sc16


@mx.compile
def _selector_scores(
    raw_logits: mx.array,
    sel_h: mx.array,
    pred_cb: mx.array,
    succ_cb: mx.array,
    anchor_tok: mx.array,
    softcap: float,
):
    """Top-16 candidates + bigram codebook scores for positions 0..n_pos-1 (Splash sampling.metal)."""
    top_idx = mx.argpartition(-raw_logits, 16, axis=-1)[:, :16]  # (P, 16)
    unary = mx.take_along_axis(raw_logits, top_idx, axis=-1).astype(mx.float32)  # (P, 16)
    return _selector_scores_from_top16(top_idx, unary, sel_h, pred_cb, succ_cb, anchor_tok, softcap)


# ---------------------------------------------------------------------------------------------------- Splash LookupIndex
class LookupIndex:
    """Bounded suffix automaton over prompt + decoded tokens (Splash runtime/model/LookupProposals.hpp)."""

    MIN_MATCH = 3
    LONGEST_MATCH = 64

    def __init__(self):
        self.tokens: list[int] = []
        self._len = [0]
        self._link = [-1]
        self._next: list[dict[int, int]] = [{}]
        self._end = [-1]
        self._last = 0
        self.match_len = 0
        self.match_end = 0

    def reset(self) -> None:
        self.tokens.clear()
        self._len = [0]
        self._link = [-1]
        self._next = [{}]
        self._end = [-1]
        self._last = 0
        self.match_len = 0
        self.match_end = 0

    def append_one(self, tok: int) -> None:
        idx = len(self.tokens)
        self.tokens.append(tok)
        cur = len(self._len)
        self._len.append(self._len[self._last] + 1)
        self._link.append(0)
        self._next.append({})
        self._end.append(idx)

        p = self._last
        nxt = self._next
        while p != -1 and tok not in nxt[p]:
            nxt[p][tok] = cur
            p = self._link[p]
        if p == -1:
            self._link[cur] = 0
        else:
            q = nxt[p][tok]
            if self._len[p] + 1 == self._len[q]:
                self._link[cur] = q
            else:
                clone = len(self._len)
                self._len.append(self._len[p] + 1)
                self._link.append(self._link[q])
                self._next.append(dict(nxt[q]))
                self._end.append(self._end[q])
                while p != -1 and nxt[p].get(tok) == q:
                    nxt[p][tok] = clone
                    p = self._link[p]
                self._link[q] = clone
                self._link[cur] = clone
        self._last = cur

        # Record match BEFORE updating ancestor end positions (previous occurrence strictly < idx).
        q = self._link[cur]
        if q > 0 and self._len[q] >= self.MIN_MATCH and self._end[q] >= 0:
            self.match_len = min(self._len[q], self.LONGEST_MATCH)
            self.match_end = self._end[q]
        else:
            self.match_len = 0
            self.match_end = 0

        # Propagate latest end position along suffix links (bounded to LONGEST_MATCH hops).
        anc = q
        for _ in range(self.LONGEST_MATCH):
            if anc <= 0:
                break
            self._end[anc] = idx
            anc = self._link[anc]

    def append(self, tokens) -> None:
        for t in tokens:
            self.append_one(int(t))

    def proposals(self, count: int = 7) -> list[int]:
        if self.match_len < self.MIN_MATCH:
            return []
        start = self.match_end + 1
        stop = min(start + count, len(self.tokens))
        return self.tokens[start:stop]

    def clone(self) -> "LookupIndex":
        cp = LookupIndex()
        cp.tokens = self.tokens.copy()
        cp._len = self._len.copy()
        cp._link = self._link.copy()
        cp._next = [dict(d) for d in self._next]
        cp._end = self._end.copy()
        cp._last = self._last
        cp.match_len = self.match_len
        cp.match_end = self.match_end
        return cp


class DFlash2StreamState:
    """Per-stream drafter KV context + LookupIndex sharing one DFlash2Drafter weight set."""

    def __init__(self, nl: int, nkv: int, hd: int, dtype: mx.Dtype = mx.bfloat16, *, allocate: bool = True):
        self.nl = int(nl)
        self.nkv = int(nkv)
        self.hd = int(hd)
        self.dtype = dtype
        self.ck: list[mx.array | None] = [None] * self.nl
        self.cv: list[mx.array | None] = [None] * self.nl
        self.n = 0
        self.lookup = LookupIndex()
        if allocate:
            self.ensure_cap(2048)

    def ensure_cap(self, need: int) -> None:
        cur = 0 if self.ck[0] is None else int(self.ck[0].shape[2])
        if need <= cur:
            return
        new_cap = max(2048, ((need + 1023) // 1024) * 1024)
        for i in range(self.nl):
            nk = mx.zeros((1, self.nkv, new_cap, self.hd), self.dtype)
            nv = mx.zeros((1, self.nkv, new_cap, self.hd), self.dtype)
            if self.ck[i] is not None and self.n > 0:
                nk[:, :, : self.n, :] = self.ck[i][:, :, : self.n, :]
                nv[:, :, : self.n, :] = self.cv[i][:, :, : self.n, :]
            self.ck[i], self.cv[i] = nk, nv

    def clone(self) -> "DFlash2StreamState":
        cp = DFlash2StreamState(self.nl, self.nkv, self.hd, self.dtype, allocate=False)
        cp.n = int(self.n)
        cp.lookup = self.lookup.clone()
        cp.ck = [None if a is None else (a + mx.zeros((), a.dtype)) for a in self.ck]
        cp.cv = [None if a is None else (a + mx.zeros((), a.dtype)) for a in self.cv]
        return cp


_DFLASH2_BATCH_MAX_C = max(1, min(8, int(os.environ.get("LOOPLING_DFLASH2_BATCH_MAX_C", "4"))))


def _default_batch_w(C: int) -> int:
    ov = os.environ.get("LOOPLING_DFLASH2_BATCH_W")
    if ov:
        return max(2, min(8, int(ov)))
    if C <= 2:
        return 3
    return 2


# ---------------------------------------------------------------------------------------------------- Splash VerifyBudget
def _logit(p: float) -> float:
    p = max(1e-6, min(1.0 - 1e-6, float(p)))
    return math.log(p / (1.0 - p))


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z)) if z >= 0.0 else math.exp(z) / (1.0 + math.exp(z))


def _solve_linear(h: list[list[float]], g: list[float]) -> list[float]:
    n = len(g)
    h = [row[:] for row in h]
    g = g[:]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(h[r][col]))
        h[col], h[piv] = h[piv], h[col]
        g[col], g[piv] = g[piv], g[col]
        diag = h[col][col]
        if abs(diag) < 1e-12:
            diag = 1e-12
        for row in range(col + 1, n):
            fac = h[row][col] / diag
            for k in range(col, n):
                h[row][k] -= fac * h[col][k]
            g[row] -= fac * g[col]
    x = [0.0] * n
    for row in range(n - 1, -1, -1):
        s = g[row]
        for k in range(row + 1, n):
            s -= h[row][k] * x[k]
        x[row] = s / (h[row][row] if abs(h[row][row]) > 1e-12 else 1e-12)
    return x


class VerifyBudget:
    """Online-calibrated acceptance estimator (Splash runtime/model/VerifyBudget.cpp)."""

    PRIOR_SLOPE = 0.80
    PRIOR_INTERCEPTS = (0.0, -0.15, -0.25, -0.35, -0.40, -0.45, -0.50)
    SLOPE_PREC = 10.0
    INTERCEPT_PREC = 1.5
    MAX_LABELS = 256
    REFIT_CYCLES = 4

    def __init__(self):
        self.slope = self.PRIOR_SLOPE
        self.intercepts = list(self.PRIOR_INTERCEPTS)
        self.labels: deque[tuple[int, float, bool]] = deque(maxlen=self.MAX_LABELS)
        self.cycles = 0

    def reset(self) -> None:
        self.slope = self.PRIOR_SLOPE
        self.intercepts = list(self.PRIOR_INTERCEPTS)
        self.labels.clear()
        self.cycles = 0

    def prob(self, pos: int, draft_prob: float) -> float:
        pos = max(0, min(6, pos))
        return _sigmoid(self.slope * _logit(draft_prob) + self.intercepts[pos])

    def observe(self, parents: tuple[int, ...], depths: tuple[int, ...], draft_probs: list[float], tokens: list[int], target_sel: list[int]) -> None:
        # Target selected target_sel[slot] at each verified slot; record labels along the accepted path.
        on_path = [False] * len(tokens)
        on_path[0] = True
        for i in range(1, len(tokens)):
            par = parents[i]
            if not on_path[par]:
                continue
            acc = tokens[i] == target_sel[par]
            if acc:
                on_path[i] = True
            q = draft_probs[i]
            if 0.0 <= q <= 1.0:
                pos = max(0, min(6, depths[i] - 1))
                self.labels.append((pos, _logit(q), acc))
        self.cycles += 1
        if self.cycles % self.REFIT_CYCLES == 0 and len(self.labels) >= 16:
            self._refit()

    def _refit(self) -> None:
        prior = [self.PRIOR_SLOPE, *self.PRIOR_INTERCEPTS]
        theta = [self.slope, *self.intercepts]
        precs = [self.SLOPE_PREC] + [self.INTERCEPT_PREC] * 7
        labs = list(self.labels)

        def obj(th):
            val = sum(0.5 * precs[i] * (th[i] - prior[i]) ** 2 for i in range(8))
            for pos, feat, acc in labs:
                z = th[0] * feat + th[1 + pos]
                m = z if acc else -z
                val += math.log1p(math.exp(-m)) if m > 0.0 else (-m + math.log1p(math.exp(m)))
            return val

        cur = obj(theta)
        for _ in range(16):
            grad = [precs[i] * (theta[i] - prior[i]) for i in range(8)]
            hess = [[0.0] * 8 for _ in range(8)]
            for i in range(8):
                hess[i][i] = precs[i]
            for pos, feat, acc in labs:
                b = 1 + pos
                z = theta[0] * feat + theta[b]
                p = _sigmoid(z)
                res = p - (1.0 if acc else 0.0)
                w = p * (1.0 - p)
                grad[0] += res * feat
                grad[b] += res
                hess[0][0] += w * feat * feat
                hess[0][b] += w * feat
                hess[b][0] += w * feat
                hess[b][b] += w
            step = _solve_linear(hess, grad)
            scale = 1.0
            moved = False
            nxt = theta
            for _ in range(20):
                cand = [theta[i] - scale * step[i] for i in range(8)]
                val = obj(cand)
                if val <= cur:
                    nxt = cand
                    cur = val
                    moved = True
                    break
                scale *= 0.5
            if not moved:
                break
            max_d = max(abs(nxt[i] - theta[i]) for i in range(8))
            theta = nxt
            if max_d < 1e-6:
                break
        self.slope = max(0.05, min(2.0, theta[0]))
        self.intercepts = [max(-4.0, min(4.0, v)) for v in theta[1:]]


# ---------------------------------------------------------------------------------------------------- Splash MoEStepCosts + AR Fallback
class MoEStepCosts:
    """2-point online MoE step cost model + AR fallback (Splash StepCosts / CostLine)."""

    def __init__(self, quant_bits: int):
        self.t_ar = 9.9
        self.t_draft = 1.7 if quant_bits else 5.5
        self.t_v2 = 12.73
        self.t_v8 = 23.14
        self._seeded_verify = False
        self.ar_cooldown = 0
        self.ar_backoff = 4
        self.low_streak = 0
        self.ema_ratio = 1.15

    def observe_ar(self, ms: float) -> None:
        if ms <= 0.0:
            return
        if not self._seeded_verify:
            self.t_ar = ms
            self.t_v2 = 1.28 * ms
            self.t_v8 = 2.32 * ms
        else:
            self.t_ar = 0.8 * self.t_ar + 0.2 * ms

    def observe_draft(self, ms: float) -> None:
        if ms > 0.0:
            self.t_draft = 0.75 * self.t_draft + 0.25 * ms

    def observe_verify(self, width: int, ms: float, keep: int, draft_ms: float) -> None:
        if ms <= 0.0 or width < 2:
            return
        u = max(0.0, min(1.0, (width - 2) / 6.0))
        if not self._seeded_verify:
            self._seeded_verify = True
            # Preserve the empirical p3 MoE width slope while shifting the baseline to observed ms.
            pred0 = self.t_v2 + u * (self.t_v8 - self.t_v2)
            shift = max(-0.4 * pred0, min(0.4 * pred0, ms - pred0))
            self.t_v2 = max(self.t_ar * 1.15, self.t_v2 + shift)
            self.t_v8 = max(self.t_v2 + 3.0, self.t_v8 + shift)
        else:
            pred = self.t_v2 + u * (self.t_v8 - self.t_v2)
            err = max(-0.5 * pred, min(0.5 * pred, ms - pred))
            lr = 0.25
            self.t_v2 = max(self.t_ar * 1.15, self.t_v2 + lr * (1.0 - u) * err)
            self.t_v8 = max(self.t_v2 + 1.0, self.t_v8 + lr * u * err)

        # Track realized vs AR rate; back off temporarily if speculative cycles lose repeatedly.
        cyc_ms = ms + draft_ms
        ratio = (keep * self.t_ar) / max(1e-3, cyc_ms)
        self.ema_ratio = 0.65 * self.ema_ratio + 0.35 * ratio
        if ratio < 0.96 and self.ema_ratio < 0.98:
            self.low_streak += 1
            if self.low_streak >= 2:
                self.ar_cooldown = self.ar_backoff
                self.ar_backoff = min(16, self.ar_backoff * 2)
                self.low_streak = 0
        elif ratio >= 1.02:
            self.low_streak = 0
            if keep >= 2:
                self.ar_backoff = 4

    def cost(self, width: int) -> float:
        if width <= 1:
            return self.t_ar
        u = max(0.0, min(1.0, (width - 2) / 6.0))
        return self.t_draft + self.t_v2 + u * (self.t_v8 - self.t_v2)

    def choose_width(self, sorted_chances: list[float], w_max: int) -> int:
        # Once _forward_block has already run, always verify at least W=2 (never discard draft).
        limit = min(w_max, 1 + len(sorted_chances))
        if limit < 2:
            return 1
        best_w = 2
        exp_toks = 1.0 + max(0.0, min(1.0, sorted_chances[0]))
        best_rate = exp_toks / max(1e-3, self.cost(2))
        for w in range(3, limit + 1):
            exp_toks += max(0.0, min(1.0, sorted_chances[w - 2]))
            rate = exp_toks / max(1e-3, self.cost(w))
            if rate > best_rate:
                best_rate = rate
                best_w = w
        return best_w


class DFlash2Drafter:
    """5-layer DFlash 2 block-diffusion drafter with sliding context KV cache."""

    def __init__(self, target, draft_dir: str = DFLASH2_DIR, quant_bits: int | None = None):
        cfg = json.load(open(os.path.join(draft_dir, "config.json")))
        dc = cfg["dflash_config"]
        self.cfg = cfg
        self.dc = dc
        self.H = int(cfg["hidden_size"])
        self.nl = int(cfg["num_hidden_layers"])
        self.nh = int(cfg["num_attention_heads"])
        self.nkv = int(cfg["num_key_value_heads"])
        self.hd = int(cfg["head_dim"])
        self.inter = int(cfg["intermediate_size"])
        self.window = int(cfg["sliding_window"])
        self.theta = float(cfg["rope_parameters"]["rope_theta"])
        self.block = int(dc["block_size"])
        self.mask_id = int(dc["mask_token_id"])
        self.rank = int(dc["selector_rank"])
        self.top_k = int(dc["selector_top_k"])
        self.softcap = float(dc.get("final_logit_softcapping", 30.0))
        self.target_layer_ids = [int(x) for x in dc["target_layer_ids"]]

        if quant_bits is None:
            quant_bits = int(os.environ.get("LOOPLING_DFLASH2_QUANT", "4"))
        self.qbits = quant_bits
        self._fused = (
            quant_bits == 4
            and self.H == 2304
            and self.hd == 128
            and os.environ.get("LOOPLING_DFLASH2_FUSED", "1") not in ("0", "", "false", "False")
        )
        self._body_mode = os.environ.get("LOOPLING_DFLASH2_BODY_MODE", "qmm")
        self._gu_block8 = self._fused and self._body_mode in ("mma1", "u2")
        self._kv_dtype = mx.bfloat16
        self._fwd_compiled: dict = {}
        self._ctx_compiled: dict = {}
        self._draft_cfg_cache: dict = {}

        w = mx.load(os.path.join(draft_dir, "model.safetensors"))
        self.fc = _quantize_if(w["fc.weight"], quant_bits)
        self.hidden_norm = w["hidden_norm.weight"]
        self.final_norm = w["norm.weight"]
        self.sel_proj = _quantize_if(w["candidate_selector.hidden_projection.weight"], quant_bits)
        self.pred_cb = w["candidate_selector.predecessor_codebook"]
        self.succ_cb = w["candidate_selector.successor_codebook"]

        self.layers = []
        to_eval = [self.fc, self.hidden_norm, self.final_norm, self.sel_proj, self.pred_cb, self.succ_cb]
        for i in range(self.nl):
            p = f"layers.{i}."
            q_w = w[p + "self_attn.q_proj.weight"]
            k_w = w[p + "self_attn.k_proj.weight"]
            v_w = w[p + "self_attn.v_proj.weight"]
            g_w = w[p + "mlp.gate_proj.weight"]
            u_w = w[p + "mlp.up_proj.weight"]
            qkv_w = mx.concatenate([q_w, k_w, v_w], axis=0)
            kv_w = mx.concatenate([k_w, v_w], axis=0)
            if self._gu_block8:
                gu_w = mx.stack([g_w.reshape(-1, 8, self.H), u_w.reshape(-1, 8, self.H)], axis=1).reshape(-1, self.H)
            else:
                gu_w = mx.concatenate([g_w, u_w], axis=0)
            lw = {
                "in_ln": w[p + "input_layernorm.weight"],
                "post_ln": w[p + "post_attention_layernorm.weight"],
                "a_base": w[p + "attention_conv.base_kernel"],
                "a_dyn": _quantize_if(w[p + "attention_conv.kernel_projection.weight"], quant_bits),
                "m_base": w[p + "mlp_conv.base_kernel"],
                "m_dyn": _quantize_if(w[p + "mlp_conv.kernel_projection.weight"], quant_bits),
                "qkv": _quantize_if(qkv_w, quant_bits),
                "kv": _quantize_if(kv_w, quant_bits),
                "q_norm": w[p + "self_attn.q_norm.weight"],
                "k_norm": w[p + "self_attn.k_norm.weight"],
                "o_proj": _quantize_if(w[p + "self_attn.o_proj.weight"], quant_bits),
                "gu": _quantize_if(gu_w, quant_bits),
                "down": _quantize_if(w[p + "mlp.down_proj.weight"], quant_bits),
            }
            self.layers.append(lw)
            to_eval.extend(lw.values())
        del w, q_w, k_w, v_w, g_w, u_w, qkv_w, kv_w, gu_w
        mx.clear_cache()

        self.target = target
        self._mask_emb = target.model.language_model.embed(mx.array([[self.mask_id]], mx.int32))[0]  # (1, H)
        self._mask_7 = mx.broadcast_to(self._mask_emb, (self.block - 1, self.H))
        lm = target.lm_head
        self._lm_aff8 = (
            getattr(lm, "mode", None) == "affine"
            and getattr(lm, "bits", None) == 8
            and getattr(lm, "group_size", None) == 64
            and "bias" not in lm
        )
        self._lm_t = (lm["weight"], lm["scales"], lm["biases"]) if self._lm_aff8 else None
        self._unpadded_vocab = int(target.args.text.unpadded_vocab_size)
        lm_qbits = int(os.environ.get("LOOPLING_DFLASH2_LM_QUANT", "4"))
        self._lm_q4 = None
        if lm_qbits == 4 and self._lm_aff8:
            w8_f = mx.dequantize(
                lm["weight"], scales=lm["scales"], biases=lm["biases"], group_size=64, bits=8, mode="affine"
            )[: self._unpadded_vocab]
            wq4, sc4, bi4 = mx.quantize(w8_f, group_size=64, bits=4, mode="affine")
            del w8_f
            self._lm_q4 = (wq4, sc4, bi4)
            to_eval.extend([wq4, sc4, bi4])
        self._df_pre_rope, self._df_post_attn = _get_df_compiled(
            self.nh, self.nkv, self.hd, self.inter, self.qbits
        )
        to_eval.extend([self._mask_emb, self._mask_7])
        mx.eval(*to_eval)
        mx.clear_cache()

        self.ck: list[mx.array | None] = [None] * self.nl
        self.cv: list[mx.array | None] = [None] * self.nl
        self.n = 0
        self.reset()

    def _lin(self, x: mx.array, w) -> mx.array:
        if self.qbits:
            wq, sc, bi = w
            return mx.quantized_matmul(
                x, wq, scales=sc, biases=bi, transpose=True, group_size=64, bits=self.qbits, mode="affine"
            )
        return x @ w

    def _lin_body(self, x: mx.array, w, use_qmv4: bool) -> mx.array:
        wq, sc, bi = w
        if use_qmv4:
            return RQ4.qmv4_bf16_ro_rows(x, wq, sc, bi)
        return mx.quantized_matmul(
            x, wq, scales=sc, biases=bi, transpose=True, group_size=64, bits=4, mode="affine"
        )

    def reset(self) -> None:
        self.ck = [None] * self.nl
        self.cv = [None] * self.nl
        self.n = 0

    def _ensure_cap(self, need: int) -> None:
        cur = 0 if self.ck[0] is None else int(self.ck[0].shape[2])
        if need <= cur:
            return
        new_cap = max(2048, ((need + 1023) // 1024) * 1024)
        dt = self._kv_dtype
        for i in range(self.nl):
            nk = mx.zeros((1, self.nkv, new_cap, self.hd), dt)
            nv = mx.zeros((1, self.nkv, new_cap, self.hd), dt)
            if self.ck[i] is not None and self.n > 0:
                nk[:, :, : self.n, :] = self.ck[i][:, :, : self.n, :]
                nv[:, :, : self.n, :] = self.cv[i][:, :, : self.n, :]
            self.ck[i], self.cv[i] = nk, nv

    def _get_append_ctx_compiled(self, n_tok: int, cap: int):
        key = (n_tok, cap)
        fn = self._ctx_compiled.get(key)
        if fn is not None:
            return fn
        H = self.H
        nl = self.nl
        nkv, hd, theta = self.nkv, self.hd, self.theta
        kw = nkv * hd
        fc = self.fc
        hidden_norm = self.hidden_norm
        layers = self.layers
        qbits = self.qbits

        def _qmm(x, w):
            if qbits:
                return mx.quantized_matmul(
                    x, w[0], scales=w[1], biases=w[2], transpose=True, group_size=64, bits=qbits, mode="affine"
                )
            return x @ w

        @mx.compile
        def _run(feats, pos_arr, *ckv):
            cks = list(ckv[:nl])
            cvs = list(ckv[nl:])
            flat = feats.reshape(n_tok, 4 * H)
            ctx = mx.fast.rms_norm(_qmm(flat, fc), hidden_norm, _EPS)
            rope_off = pos_arr[0]
            out_k = []
            out_v = []
            for i in range(nl):
                lw = layers[i]
                kv = _qmm(ctx, lw["kv"])
                k = mx.fast.rms_norm(kv[:, :kw].reshape(1, n_tok, nkv, hd), lw["k_norm"], _EPS).transpose(0, 2, 1, 3)
                k = mx.fast.rope(k, hd, traditional=False, base=theta, scale=1.0, offset=rope_off)
                v = kv[:, kw:].reshape(1, n_tok, nkv, hd).transpose(0, 2, 1, 3)
                out_k.append(mx.slice_update(cks[i], k, pos_arr, axes=(2,)))
                out_v.append(mx.slice_update(cvs[i], v, pos_arr, axes=(2,)))
            return tuple(out_k) + tuple(out_v)

        self._ctx_compiled[key] = _run
        return _run

    def append_ctx_gpu(self, feats: mx.array, pos_arr: mx.array) -> None:
        """Project target taps `(1, N, 4, H)` -> context K/V at GPU `pos_arr (1,) i32` (zero sync)."""
        n_tok = int(feats.shape[1])
        if n_tok == 0:
            return
        if self._fused and n_tok <= 16 and self.ck[0] is not None:
            cap = int(self.ck[0].shape[2])
            fn = self._get_append_ctx_compiled(n_tok, cap)
            cks = self.ck
            cvs = self.cv
            self.ck = [None] * self.nl
            self.cv = [None] * self.nl
            res = fn(feats, pos_arr, *cks, *cvs)
            self.ck = list(res[: self.nl])
            self.cv = list(res[self.nl :])
            return
        flat = feats.reshape(n_tok, 4 * self.H)
        ctx = _rms(self._lin(flat, self.fc), self.hidden_norm)  # (N, H)
        rope_off = pos_arr[0]
        kw = self.nkv * self.hd
        for i, lw in enumerate(self.layers):
            kv = self._lin(ctx, lw["kv"])  # (N, 2*nkv*hd)
            k = kv[:, :kw].reshape(1, n_tok, self.nkv, self.hd)
            v = kv[:, kw:].reshape(1, n_tok, self.nkv, self.hd)
            k = _rms(k, lw["k_norm"]).transpose(0, 2, 1, 3)
            k = mx.fast.rope(k, self.hd, traditional=False, base=self.theta, scale=1.0, offset=rope_off)
            v = v.transpose(0, 2, 1, 3)
            ck_i = self.ck[i]
            self.ck[i] = None
            self.ck[i] = mx.slice_update(ck_i, k, pos_arr, axes=(2,))
            cv_i = self.cv[i]
            self.cv[i] = None
            self.cv[i] = mx.slice_update(cv_i, v, pos_arr, axes=(2,))

    def append_ctx(self, feats: mx.array) -> None:
        """Project target taps `(1, N, 4, H)` -> context K/V for all 5 draft layers."""
        n_tok = int(feats.shape[1])
        if n_tok == 0:
            return
        self._ensure_cap(self.n + n_tok + self.block)
        self.append_ctx_gpu(feats, FD._wpos(self.n))
        self.n += n_tok

    def _ctx_slice(self, i: int):
        if self.n == 0:
            return None, None
        start = max(0, self.n - self.window)
        return self.ck[i][:, :, start : self.n, :], self.cv[i][:, :, start : self.n, :]

    def _get_fwd_bf16_compiled(self, n_pos: int, cap: int, body_mode: str, use_fuse_top16: bool, top16_cfg: str, sdpa_q2: bool):
        key = (n_pos, cap, body_mode, use_fuse_top16, top16_cfg, sdpa_q2)
        fn = self._fwd_compiled.get(key)
        if fn is not None:
            return fn
        scale = self.hd ** -0.5
        layers = self.layers
        nl = self.nl
        nh, nkv, hd, theta, window = self.nh, self.nkv, self.hd, self.theta, self.window
        mask_7 = self._mask_7
        final_norm = self.final_norm
        sel_proj = self.sel_proj
        lm_q4 = self._lm_q4
        lm_aff8 = self._lm_aff8
        lm_t = self._lm_t
        unpadded_vocab = self._unpadded_vocab
        pred_cb, succ_cb, softcap = self.pred_cb, self.succ_cb, self.softcap
        embed = self.target.model.language_model.embed
        target_lm = self.target.lm_head

        gu_block8 = self._gu_block8
        inter = self.inter

        def _lin_b(x, w):
            wq, sc, bi = w
            if body_mode == "mma1" and int(x.shape[0]) == 8:
                return RQ4.qmv4_bf16_mma8_rows(x, wq, sc, bi)
            if body_mode == "u2":
                return RQ4.qmv4_bf16_u2_rows(x, wq, sc, bi)
            if body_mode in ("ro", "mma1"):
                return RQ4.qmv4_bf16_ro_rows(x, wq, sc, bi)
            return mx.quantized_matmul(
                x, wq, scales=sc, biases=bi, transpose=True, group_size=64, bits=4, mode="affine"
            )

        def _swiglu_b(x, w):
            wq, sc, bi = w
            if gu_block8:
                if body_mode == "mma1" and int(x.shape[0]) == 8:
                    return RQ4.qmv4_bf16_mma8_swiglu_rows(x, wq, sc, bi)
                return RQ4.qmv4_bf16_u2_swiglu_b8_rows(x, wq, sc, bi)
            gu = _lin_b(x, w)
            return nn.silu(gu[:, :inter]) * gu[:, inter:]

        @mx.compile
        def _run(anchor_tok, pos_arr, *ckv):
            cks = ckv[:nl]
            cvs = ckv[nl:]
            anc_emb = embed(anchor_tok.reshape(1, 1))[0]
            x = mx.concatenate([anc_emb, mask_7], axis=0)
            h_norm = mx.fast.rms_norm(x, layers[0]["in_ln"], _EPS)
            for i in range(nl):
                lw = layers[i]
                dyn_a = _lin_b(h_norm, lw["a_dyn"])
                c_in = RQ4.dyn_conv_stage0_bf16(h_norm, dyn_a, lw["a_base"])
                qkv = _lin_b(c_in, lw["qkv"])
                q, k, v = RQ4.qkv_norm_rope_bf16(qkv, lw["q_norm"], lw["k_norm"], pos_arr, nh, nkv, hd, theta)
                attn = RQ4.sdpa_draft_bf16_gpu(q, cks[i], cvs[i], k, v, pos_arr, scale=scale, window=window)
                proj = _lin_b(attn, lw["o_proj"])
                res, m_norm = RQ4.dyn_conv_res_rms_bf16(proj, dyn_a, lw["a_base"], x, lw["post_ln"])
                dyn_m = _lin_b(m_norm, lw["m_dyn"])
                m_in = RQ4.dyn_conv_stage0_bf16(m_norm, dyn_m, lw["m_base"])
                act = _swiglu_b(m_in, lw["gu"])
                m_out = _lin_b(act, lw["down"])
                next_ln = layers[i + 1]["in_ln"] if i + 1 < nl else final_norm
                x, h_norm = RQ4.dyn_conv_res_rms_bf16(m_out, dyn_m, lw["m_base"], res, next_ln)

            hid = h_norm[1 : 1 + n_pos]
            sel_h = _lin_b(hid, sel_proj).astype(mx.float32)
            if lm_q4 is not None:
                wq4, sc4, bi4 = lm_q4
                if use_fuse_top16:
                    top_idx, unary = RQ4.qmv4_bf16_top16_rows(hid, wq4, sc4, bi4)
                    return _selector_scores_from_top16(
                        top_idx, unary, sel_h, pred_cb, succ_cb, anchor_tok, softcap
                    )
                raw_logits = RQ4.qmv4_bf16_ro_rows(hid, wq4, sc4, bi4)
            elif lm_aff8:
                raw_logits = RQ.qmv_rows(hid, *lm_t)[:, :unpadded_vocab]
            else:
                raw_logits = target_lm(hid)[:, :unpadded_vocab]
            return _selector_scores(raw_logits, sel_h, pred_cb, succ_cb, anchor_tok, softcap)

        self._fwd_compiled[key] = _run
        return _run

    def _forward_block_fused(self, anchor_tok: mx.array, n_pos: int = 7, pos_arr: mx.array | None = None):
        if self.ck[0] is None:
            self._ensure_cap(1)
        n_pos = max(2, min(self.block - 1, int(n_pos)))
        cap = int(self.ck[0].shape[2])
        body_mode = os.environ.get("LOOPLING_DFLASH2_BODY_MODE", self._body_mode)
        use_fuse_top16 = os.environ.get("LOOPLING_DFLASH2_FUSE_TOP16", "1") not in ("0", "", "false", "False")
        top16_cfg = (
            os.environ.get("LOOPLING_DFLASH2_TOP16_MODE", "mma")
            + "|"
            + os.environ.get("LOOPLING_DFLASH2_TOP16_MMA_CFG", "")
            + "|"
            + os.environ.get("LOOPLING_DFLASH2_TOP16_CFG", "")
            + "|"
            + os.environ.get("LOOPLING_DFLASH2_MMA_NSG", "")
        )
        sdpa_q2 = os.environ.get("LOOPLING_DFLASH2_SDPA_Q2", "0") not in ("0", "", "false", "False")
        fn = self._get_fwd_bf16_compiled(n_pos, cap, body_mode, use_fuse_top16, top16_cfg, sdpa_q2)
        if pos_arr is None:
            pos_arr = FD._wpos(self.n)
        anc = anchor_tok.reshape(1)
        if anc.dtype != mx.uint32:
            anc = anc.astype(mx.uint32)
        return fn(anc, pos_arr, *self.ck, *self.cv)

    def draft_and_pack_gpu(
        self,
        anchor_tok: mx.array,
        w_max: int,
        *,
        pos_arr: mx.array | None = None,
        use_tree: bool = False,
        cost_arr: mx.array | None = None,
        prior_cfg: mx.array | None = None,
    ):
        """Zero-sync DFlash2 forward + chain/tree packer on GPU."""
        w_max = max(2, min(self.block, int(w_max)))
        top_idx, sc0, sc16 = self._forward_block_fused(anchor_tok, n_pos=w_max - 1, pos_arr=pos_arr)
        if use_tree:
            return RQ4.dflash2_pack_tree_dink_gpu(top_idx, sc0, sc16, anchor_tok, w_max, cost_arr, prior_cfg)
        return RQ4.dflash2_pack_chain_gpu(top_idx, sc0, sc16, anchor_tok, w_max)

    def draft_and_pack_coupled_gpu(
        self,
        anchor_tok: mx.array,
        w_max: int,
        u_draft: mx.array,
        draft_temp: float,
        *,
        pos_arr: mx.array | None = None,
    ):
        """Zero-sync DFlash2 forward + Leviathan coupled stochastic chain packer on GPU."""
        w_max = max(2, min(self.block, int(w_max)))
        top_idx, sc0, sc16 = self._forward_block_fused(anchor_tok, n_pos=w_max - 1, pos_arr=pos_arr)
        inv_dt = 1.0 / max(float(draft_temp), 1e-6)
        dcfg = self._draft_cfg_cache.get(inv_dt)
        if dcfg is None:
            dcfg = self._draft_cfg_cache[inv_dt] = mx.array([inv_dt], mx.float32)
        return RQ4.dflash2_pack_coupled_chain_gpu(top_idx, sc0, sc16, anchor_tok, w_max, u_draft, dcfg)

    def _forward_block(self, anchor_tok: mx.array, n_pos: int = 7):
        if self._fused:
            return self._forward_block_fused(anchor_tok, n_pos=n_pos)
        anc_emb = self.target.model.language_model.embed(anchor_tok.reshape(1, 1))[0]  # (1, H)
        x = mx.concatenate([anc_emb, self._mask_7], axis=0)  # (8, H)
        B = self.block
        scale = self.hd ** -0.5
        offset = self.n
        pre_rope = self._df_pre_rope
        post_attn = self._df_post_attn

        for i, lw in enumerate(self.layers):
            q, k, v, dyn_a = pre_rope(
                x, lw["in_ln"], lw["a_dyn"], lw["a_base"], lw["qkv"], lw["q_norm"], lw["k_norm"]
            )
            q = mx.fast.rope(q, self.hd, traditional=False, base=self.theta, scale=1.0, offset=offset)
            k = mx.fast.rope(k, self.hd, traditional=False, base=self.theta, scale=1.0, offset=offset)
            ck, cv = self._ctx_slice(i)
            k_all = k if ck is None else mx.concatenate([ck, k], axis=2)
            v_all = v if cv is None else mx.concatenate([cv, v], axis=2)
            attn = mx.fast.scaled_dot_product_attention(q, k_all, v_all, scale=scale)
            attn = attn.transpose(0, 2, 1, 3).reshape(B, self.nh * self.hd)
            x = post_attn(
                x, attn, lw["o_proj"], dyn_a, lw["a_base"], lw["post_ln"], lw["m_dyn"], lw["m_base"], lw["gu"], lw["down"]
            )

        n_pos = max(2, min(self.block - 1, int(n_pos)))
        hid = _rms(x, self.final_norm)[1 : 1 + n_pos]  # (n_pos, H)
        if self._lm_q4 is not None:
            wq4, sc4, bi4 = self._lm_q4
            if _USE_QMV4 and hid.dtype == mx.bfloat16 and 1 <= hid.shape[0] <= 16 and hid.shape[1] % 256 == 0 and sc4.dtype == mx.bfloat16:
                raw_logits = RQ4.qmv4_rows(hid, wq4, sc4, bi4)
            else:
                raw_logits = mx.quantized_matmul(
                    hid, wq4, scales=sc4, biases=bi4, transpose=True, group_size=64, bits=4, mode="affine"
                )
        elif self._lm_aff8:
            raw_logits = RQ.qmv_rows(hid, *self._lm_t)[:, : self._unpadded_vocab]
        else:
            raw_logits = self.target.lm_head(hid)[:, : self._unpadded_vocab]
        sel_h = self._lin(hid, self.sel_proj).astype(mx.float32)  # (n_pos, R)
        return _selector_scores(raw_logits, sel_h, self.pred_cb, self.succ_cb, anchor_tok, self.softcap)

    def draft_chain(self, anchor_tok: mx.array, width: int = 8, prefetched=None) -> mx.array:
        """Return `[anchor, d_1 .. d_{width-1}]` `(width,)` int32 via greedy selector chain."""
        width = max(2, min(self.block, int(width)))
        top_idx, sc0, sc16 = prefetched if prefetched is not None else self._forward_block(anchor_tok, n_pos=width - 1)
        mx.eval(top_idx, sc0, sc16)
        tl = top_idx.tolist()
        s0 = sc0.tolist()
        s16 = sc16.tolist()
        slot = max(range(len(s0)), key=s0.__getitem__)
        toks = [int(anchor_tok.item()), int(tl[0][slot])]
        for p in range(1, width - 1):
            row = s16[p - 1][slot]
            slot = max(range(len(row)), key=row.__getitem__)
            toks.append(int(tl[p][slot]))
        return mx.array(toks, mx.int32)

    def draft_hybrid_tree(
        self,
        anchor_tok: mx.array,
        w_max: int,
        budget: VerifyBudget | None = None,
        step_costs: MoEStepCosts | None = None,
        lookup_props: list[int] | None = None,
        lookup_len: int = 0,
        use_tree: bool = True,
        prefetched=None,
    ):
        """Return `(toks (W,) i32, parents, depths, raw_probs)` or `(None, ...)` when StepCosts picks W=1."""
        w_max = max(2, min(self.block, int(w_max)))
        top_idx, sc0, sc16 = prefetched if prefetched is not None else self._forward_block(anchor_tok, n_pos=w_max - 1)
        p0 = mx.softmax(sc0, axis=-1)
        p16 = mx.softmax(sc16, axis=-1)
        mx.eval(top_idx, p0, p16)
        tl = top_idx.tolist()
        p0_l = p0.tolist()
        p16_l = p16.tolist()
        anc_int = int(anchor_tok.item())

        def cal(depth: int, raw_q: float) -> float:
            if budget is None:
                return raw_q
            return budget.prob(depth - 1, raw_q)

        # Build candidate tree: each node = (neg_joint_cal_p, depth, order_id, parent_oid, slot, tok, raw_q)
        nodes = [(-1.0, 0, 0, -1, 0, anc_int, -1.0)]
        pq = []
        fan0 = 1 if not use_tree else min(4, w_max - 1)
        order0 = sorted(range(len(p0_l)), key=lambda j: -p0_l[j])[:fan0]
        for r, s in enumerate(order0):
            q = p0_l[s]
            c = cal(1, q)
            heapq.heappush(pq, (-c, 1, r, 0, s, int(tl[0][s]), q))

        # Overlay LookupProposals along the greedy lookup path from root (100% bit-exact verify).
        lprops = lookup_props[: w_max - 1] if (lookup_props and lookup_len >= LookupIndex.MIN_MATCH) else []
        if lprops:
            l_base = 0.90 if lookup_len >= 8 else (0.80 if lookup_len >= 4 else 0.65)
            cand0 = {int(tl[0][s]): (s, p0_l[s]) for s in range(len(p0_l))}
            lt0 = lprops[0]
            if lt0 in cand0:
                s0, q0 = cand0[lt0]
                boost0 = max(cal(1, q0), l_base)
                heapq.heappush(pq, (-boost0, 1, 99, 0, s0, lt0, q0))
            else:
                # Negative slot marks pure lookup node (raw_q = -1 -> excluded from neural regression labels).
                heapq.heappush(pq, (-l_base, 1, 99, 0, -1, lt0, -1.0))

        seen_edges: set[tuple[int, int]] = set()
        next_oid = 1
        lookup_chain_parent = 0
        lookup_chain_p = 1.0
        if lprops:
            l_base = 0.90 if lookup_len >= 8 else (0.80 if lookup_len >= 4 else 0.65)

        while pq and len(nodes) < w_max * 2:
            neg_jp, dep, _, par_oid, slot, tok, rq = heapq.heappop(pq)
            if (par_oid, tok) in seen_edges:
                continue
            seen_edges.add((par_oid, tok))
            cur_oid = next_oid
            next_oid += 1
            nodes.append((neg_jp, dep, cur_oid, par_oid, slot, tok, rq))

            # Advance lookup chain if this node extended the prefix of lprops
            if lprops and par_oid == lookup_chain_parent and dep <= len(lprops) and tok == lprops[dep - 1]:
                lookup_chain_parent = cur_oid
                lookup_chain_p = -neg_jp
                if dep < len(lprops):
                    nxt_lt = lprops[dep]
                    nxt_prior = lookup_chain_p * l_base * (0.94 ** dep)
                    if slot >= 0 and dep < len(p16_l) + 1:
                        cand_row = {int(tl[dep][sc]): (sc, p16_l[dep - 1][slot][sc]) for sc in range(len(p16_l[dep - 1][slot]))}
                        if nxt_lt in cand_row:
                            sc, rq2 = cand_row[nxt_lt]
                            jp2 = max(nxt_prior, lookup_chain_p * cal(dep + 1, rq2))
                            heapq.heappush(pq, (-jp2, dep + 1, 99, cur_oid, sc, nxt_lt, rq2))
                        else:
                            heapq.heappush(pq, (-nxt_prior, dep + 1, 99, cur_oid, -1, nxt_lt, -1.0))
                    else:
                        heapq.heappush(pq, (-nxt_prior, dep + 1, 99, cur_oid, -1, nxt_lt, -1.0))

            if slot >= 0 and dep < w_max - 1 and dep - 1 < len(p16_l):
                row_p = p16_l[dep - 1][slot]
                fan = 1 if not use_tree else 3
                order_c = sorted(range(len(row_p)), key=lambda j: -row_p[j])[:fan]
                for r, sc in enumerate(order_c):
                    rq2 = row_p[sc]
                    jp2 = (-neg_jp) * cal(dep + 1, rq2)
                    heapq.heappush(pq, (-jp2, dep + 1, r, cur_oid, sc, int(tl[dep][sc]), rq2))

        if len(nodes) < 2:
            return None, (), (), []

        sel_full = nodes[:w_max]
        chances = [-nd[0] for nd in sel_full[1:]]
        if step_costs is not None:
            w_opt = step_costs.choose_width(chances, w_max)
            if w_opt <= 1:
                return None, (), (), []
            sel = sel_full[:w_opt]
        else:
            sel = sel_full

        sel_sorted = [sel[0]] + sorted(sel[1:], key=lambda x: (x[1], x[2]))
        old_to_new = {x[2]: idx for idx, x in enumerate(sel_sorted)}
        block_toks = [x[5] for x in sel_sorted]
        parents = tuple(-1 if i == 0 else old_to_new[x[3]] for i, x in enumerate(sel_sorted))
        depths = tuple(x[1] for x in sel_sorted)
        raw_probs = [x[6] for x in sel_sorted]
        return mx.array(block_toks, mx.int32), parents, depths, raw_probs

    def draft_tree(self, anchor_tok: mx.array, width: int = 8, max_depth: int | None = None):
        """Legacy unweighted best-first tree helper (used when LOOPLING_DFLASH2_BUDGET=0)."""
        toks, parents, depths, _ = self.draft_hybrid_tree(
            anchor_tok, w_max=width, budget=None, step_costs=None, lookup_props=None, use_tree=True
        )
        return toks, parents, depths

    def new_stream_state(self, *, allocate: bool = True) -> DFlash2StreamState:
        return DFlash2StreamState(self.nl, self.nkv, self.hd, self._kv_dtype, allocate=allocate)

    def append_ctx_batch(self, states: list[DFlash2StreamState], feats_per_stream: list[mx.array]) -> None:
        """Project target taps for C streams in one batched fc + 5*kv matmul and scatter into each stream state."""
        ks = [int(f.shape[1]) if f.ndim == 4 else int(f.shape[0]) for f in feats_per_stream]
        tot = sum(ks)
        if tot == 0:
            return
        for st, kc in zip(states, ks):
            if kc > 0:
                st.ensure_cap(st.n + kc + self.block)
        flat = mx.concatenate(
            [f.reshape(kc, 4 * self.H) for f, kc in zip(feats_per_stream, ks) if kc > 0], axis=0
        )
        ctx = _rms(self._lin(flat, self.fc), self.hidden_norm)
        kw = self.nkv * self.hd
        offs0 = [st.n for st in states]
        for i, lw in enumerate(self.layers):
            kv = self._lin(ctx, lw["kv"])
            k_all = _rms(kv[:, :kw].reshape(tot, self.nkv, self.hd), lw["k_norm"])
            v_all = kv[:, kw:].reshape(tot, self.nkv, self.hd)
            pos = 0
            for c_idx, (st, kc) in enumerate(zip(states, ks)):
                if kc <= 0:
                    continue
                k_c = k_all[pos : pos + kc].reshape(1, kc, self.nkv, self.hd).transpose(0, 2, 1, 3)
                k_c = mx.fast.rope(k_c, self.hd, traditional=False, base=self.theta, scale=1.0, offset=offs0[c_idx])
                v_c = v_all[pos : pos + kc].reshape(1, kc, self.nkv, self.hd).transpose(0, 2, 1, 3)
                ck_i = st.ck[i]
                st.ck[i] = None
                st.ck[i] = mx.slice_update(ck_i, k_c, FD._wpos(offs0[c_idx]), axes=(2,))
                cv_i = st.cv[i]
                st.cv[i] = None
                st.cv[i] = mx.slice_update(cv_i, v_c, FD._wpos(offs0[c_idx]), axes=(2,))
                pos += kc
        for st, kc in zip(states, ks):
            st.n += kc

    def draft_batch_chains(
        self,
        states: list[DFlash2StreamState],
        anchors_c: list[int] | mx.array,
        W: int,
        *,
        use_lookup: bool = True,
    ) -> mx.array:
        """Draft `(C, W)` int32 greedy chains for C active streams sharing one batched drafter forward."""
        W = max(2, min(self.block, int(W)))
        C = len(states)
        if isinstance(anchors_c, mx.array):
            anc_list = anchors_c.tolist()
            anc_arr_all = anchors_c.reshape(C).astype(mx.int32)
        else:
            anc_list = [int(a) for a in anchors_c]
            anc_arr_all = mx.array(anc_list, mx.int32)

        out_rows: list[list[int] | None] = [None] * C
        nn_idx: list[int] = []
        for c, st in enumerate(states):
            lp = st.lookup
            if use_lookup and _LOOKUP_BYPASS_MIN > 0 and lp.match_len >= _LOOKUP_BYPASS_MIN:
                props = lp.proposals(W - 1)
                if len(props) >= W - 1:
                    out_rows[c] = [anc_list[c]] + props[: W - 1]
                    continue
            nn_idx.append(c)

        if not nn_idx:
            return mx.array(out_rows, mx.int32)

        Cn = len(nn_idx)
        anc_nn = mx.array([anc_list[c] for c in nn_idx], mx.int32)
        anc_embs = self.target.model.language_model.embed(anc_nn.reshape(Cn, 1))[:, 0, :]  # (Cn, H)
        m7 = mx.broadcast_to(self._mask_7[None, :, :], (Cn, self.block - 1, self.H))
        x = mx.concatenate([anc_embs[:, None, :], m7], axis=1)  # (Cn, 8, H)
        B_tot = Cn * self.block
        scale = self.hd ** -0.5

        ctx_lens = [max(0, states[c].n - max(0, states[c].n - self.window)) for c in nn_idx]
        ctx_starts = [max(0, states[c].n - self.window) for c in nn_idx]
        ctx_max = max(ctx_lens) if ctx_lens else 0

        attn_mask = None
        if ctx_max > 0 and any(cl != ctx_max for cl in ctx_lens):
            m_np = np.zeros((Cn, 1, self.block, ctx_max + self.block), dtype=np.float32)
            for idx_c, cl in enumerate(ctx_lens):
                if cl < ctx_max:
                    m_np[idx_c, :, :, cl:ctx_max] = -1e9
            attn_mask = mx.array(m_np).astype(mx.bfloat16)

        offs_arr = mx.array([states[c].n for c in nn_idx], mx.int32)
        pre_rope_b, post_attn_b, sel_n1 = _get_df_batch_compiled(
            self.nh, self.nkv, self.hd, self.inter, self.qbits, self.theta
        )
        pads = [
            mx.zeros((1, self.nkv, ctx_max - cl, self.hd), self._kv_dtype) if (ctx_max > 0 and cl < ctx_max) else None
            for cl in ctx_lens
        ]
        for i, lw in enumerate(self.layers):
            q_cur, k_cur, v_cur, dyn_a = pre_rope_b(
                x, lw["in_ln"], lw["a_dyn"], lw["a_base"], lw["qkv"], lw["q_norm"], lw["k_norm"], offs_arr
            )
            if ctx_max > 0:
                ks_ctx, vs_ctx = [], []
                for idx_c, c in enumerate(nn_idx):
                    cl = ctx_lens[idx_c]
                    s0 = ctx_starts[idx_c]
                    kc = states[c].ck[i][:, :, s0 : states[c].n, :]
                    vc = states[c].cv[i][:, :, s0 : states[c].n, :]
                    pad = pads[idx_c]
                    if pad is not None:
                        kc = mx.concatenate([kc, pad], axis=2)
                        vc = mx.concatenate([vc, pad], axis=2)
                    ks_ctx.append(kc)
                    vs_ctx.append(vc)
                k_all = mx.concatenate([mx.concatenate(ks_ctx, axis=0), k_cur], axis=2)
                v_all = mx.concatenate([mx.concatenate(vs_ctx, axis=0), v_cur], axis=2)
            else:
                k_all, v_all = k_cur, v_cur
            attn = mx.fast.scaled_dot_product_attention(q_cur, k_all, v_all, scale=scale, mask=attn_mask)
            attn = attn.transpose(0, 2, 1, 3).reshape(B_tot, self.nh * self.hd)
            x = post_attn_b(
                x, attn, lw["o_proj"], dyn_a, lw["a_base"], lw["post_ln"], lw["m_dyn"], lw["m_base"], lw["gu"], lw["down"]
            )

        n_pos = W - 1
        hid = _rms(x.reshape(B_tot, self.H), self.final_norm).reshape(Cn, self.block, self.H)[:, 1 : 1 + n_pos, :]
        hid_flat = hid.reshape(Cn * n_pos, self.H)
        if self._lm_q4 is not None:
            wq4, sc4, bi4 = self._lm_q4
            raw_logits = mx.quantized_matmul(
                hid_flat, wq4, scales=sc4, biases=bi4, transpose=True, group_size=64, bits=4, mode="affine"
            )
        elif self._lm_aff8:
            raw_logits = RQ.qmv_rows(hid_flat, *self._lm_t)[:, : self._unpadded_vocab]
        else:
            raw_logits = self.target.lm_head(hid_flat)[:, : self._unpadded_vocab]
        sel_h = self._lin(hid_flat, self.sel_proj).astype(mx.float32).reshape(Cn, n_pos, self.rank)

        if n_pos == 1:
            nxt_tok = sel_n1(raw_logits, sel_h[:, 0], self.pred_cb, self.succ_cb, anc_nn, self.softcap)
            mx.eval(nxt_tok)
            for idx_c, nt in zip(nn_idx, nxt_tok.tolist()):
                out_rows[idx_c] = [anc_list[idx_c], int(nt)]
        else:
            top_idx = mx.argpartition(-raw_logits, 16, axis=-1)[:, :16]
            unary = mx.take_along_axis(raw_logits, top_idx, axis=-1).astype(mx.float32)
            top_idx_c = top_idx.reshape(Cn, n_pos, 16)
            unary_c = unary.reshape(Cn, n_pos, 16)
            sc0_list, sc16_list = [], []
            for idx_c in range(Cn):
                _, sc0_i, sc16_i = _selector_scores_from_top16(
                    top_idx_c[idx_c], unary_c[idx_c], sel_h[idx_c], self.pred_cb, self.succ_cb, anc_nn[idx_c], self.softcap,
                )
                sc0_list.append(sc0_i)
                sc16_list.append(sc16_i)
            sc0_all = mx.stack(sc0_list, axis=0)
            sc16_all = mx.stack(sc16_list, axis=0)
            mx.eval(top_idx_c, sc0_all, sc16_all)
            tl_all = top_idx_c.tolist()
            s0_all = sc0_all.tolist()
            s16_all = sc16_all.tolist()
            for idx_c, c in enumerate(nn_idx):
                s0 = s0_all[idx_c]
                s16 = s16_all[idx_c]
                tl = tl_all[idx_c]
                slot = max(range(len(s0)), key=s0.__getitem__)
                row = [anc_list[c], int(tl[0][slot])]
                for p in range(1, n_pos):
                    r_p = s16[p - 1][slot]
                    slot = max(range(len(r_p)), key=r_p.__getitem__)
                    row.append(int(tl[p][slot]))
                out_rows[c] = row
        # Overlay short LookupIndex proposals (len >= 3) if available? Wait: greedy lookup wins when >= MIN_MATCH?
        # Keep pure neural chain unless >= _LOOKUP_BYPASS_MIN, matching draft_hybrid_tree lookup_bypass gating.
        return mx.array(out_rows, mx.int32)


_DF2_INST: dict = {}


def _get_drafter(model, draft_dir: str = DFLASH2_DIR) -> DFlash2Drafter:
    lm_w = model.lm_head["weight"]
    qbits = int(os.environ.get("LOOPLING_DFLASH2_QUANT", "4"))
    fused = qbits == 4 and os.environ.get("LOOPLING_DFLASH2_FUSED", "1") not in ("0", "", "false", "False")
    gu_b8 = fused and os.environ.get("LOOPLING_DFLASH2_BODY_MODE", "qmm") in ("mma1", "u2")
    key = (id(model), draft_dir, qbits, fused, gu_b8)
    entry = _DF2_INST.get(key)
    if entry is not None and entry[0] is lm_w:
        dr = entry[1]
        dr.reset()
        return dr
    dr = DFlash2Drafter(model, draft_dir=draft_dir, quant_bits=qbits)
    _DF2_INST[key] = (lm_w, dr)
    return dr


def dflash2_exact_generate_step(
    prompt: mx.array,
    model,
    *,
    max_tokens: int = 256,
    sampler=None,
    prefill_step_size: int = 2048,
    prompt_cache=None,
    draft_dir: str | None = None,
    full_prompt: list[int] | None = None,
    eos_token_ids=None,
    checkpoint_callback=None,
    segment_ends: list[tuple[int, str]] | None = None,
    cache_state: dict | None = None,
    prompt_progress_callback=None,
    **kw,
):
    """Bit-exact causal-verify speculative decoding with Splash DFlash 2 drafter."""
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.models.inkling2.batchdecode import BatchDecoder, _cw_ar_cache, _std_sampler_params, build_tree_tables
    from mlx_lm.models.inkling2.mtp import _feats_from_dict, default_taps, feats_from_buf, parse_taps

    if draft_dir is None:
        draft_dir = DFLASH2_DIR
    tm = model.model.language_model
    passes = tm.args.loop_config["num_passes"]
    taps = parse_taps(os.environ.get("LOOPLING_MTP_TAPS", default_taps(passes)), passes)
    tap_keys = [t if t in ("F", "f", "o") else int(t[1:]) for t in taps]
    tap_set = {k for k in tap_keys if isinstance(k, int)}
    dr = _get_drafter(model, draft_dir)
    cache = prompt_cache if prompt_cache is not None else make_prompt_cache(model)

    stats = getattr(model, "dflash2_stats", None)
    if stats is None:
        stats = model.dflash2_stats = {"cycles": 0, "accepted": 0, "tokens": 0, "draft_ms": 0.0, "verify_ms": 0.0}
    stats.update(cycles=0, accepted=0, tokens=0, draft_ms=0.0, verify_ms=0.0)

    n_p = int(prompt.size)
    pos = 0
    if prompt_progress_callback is not None:
        prompt_progress_callback(0, n_p)
    seg_idx = 0
    seg_list = [s for s in (segment_ends or []) if 0 < s[0] < n_p]
    if n_p > 1:
        tm.tap_visits = tap_set
        while n_p - pos > 1:
            tm.tap_buf = {}
            next_seg = seg_list[seg_idx][0] if seg_idx < len(seg_list) else (n_p - 1)
            m = min(prefill_step_size, next_seg - pos)
            _ = model(prompt[pos : pos + m][None], cache=cache)
            f = feats_from_buf(tm, taps)
            dr.append_ctx(f)
            mx.eval([c.state for c in cache], dr.ck, dr.cv)
            pos += m
            if prompt_progress_callback is not None:
                prompt_progress_callback(pos, n_p)
            hit_seg = None
            if seg_idx < len(seg_list) and pos >= seg_list[seg_idx][0]:
                hit_seg = seg_list[seg_idx][1]
                seg_idx += 1
            if checkpoint_callback is not None:
                if hit_seg is not None:
                    checkpoint_callback(pos, n_p, cache, seg_type=hit_seg)
                else:
                    checkpoint_callback(pos, n_p, cache)
            mx.clear_cache()
        tm.tap_visits = None
        tm.tap_buf = {}

    dec = BatchDecoder(model, [cache])
    if not dec._b1_fast:
        raise RuntimeError("dflash2_exact_generate_step requires BatchDecoder._b1_fast")

    y, tap0 = dec.step_with_taps(prompt[pos:], tap_visits=tap_set, sampler=sampler)
    f0 = _feats_from_dict(tap0, tap_keys)
    dr.append_ctx(f0)
    mx.eval(y, dr.ck, dr.cv, *dec.kv_eval_targets(), dec.conv)
    if prompt_progress_callback is not None:
        prompt_progress_callback(n_p, n_p)

    committed_tokens: list[int] = []
    yielded_tokens: list[int] = []
    pa_queue: deque = deque()
    last_popped_rb = None

    def _read_cpu_kv():
        if dec._ring_visits:
            kv_r = dec.caches[0][dec._ring_visits[0]][0]
            off_r, idx_r = int(kv_r.offset), int(kv_r._idx)
        else:
            off_r, idx_r = 0, 0
        off_f = int(dec.caches[0][dec._full_visits[0]][0].offset) if dec._full_visits else 0
        return len(committed_tokens), off_r, idx_r, off_f

    def _snap_rb(slot_r_expr, w_slots: int, need_ring: bool = True):
        conv_bak = [list(cs) for cs in dec.conv]
        pos_bak = ring_bak = None
        if need_ring and dec._kv_stacked and dec._kv_ring is not None and dec._ring_visits:
            max_size_r = int(dec.caches[0][dec._ring_visits[0]][0].max_size)
            ar = _cw_ar_cache.get(w_slots)
            if ar is None:
                ar = _cw_ar_cache[w_slots] = mx.arange(w_slots, dtype=mx.int32)
            pos_bak = (slot_r_expr + ar) % max_size_r
            ring_bak = dec._kv_ring[:, :, :, pos_bak, :]
        return conv_bak, pos_bak, ring_bak

    def _restore_kv_slices(conv_bak, pos_bak, ring_bak):
        if conv_bak is not None:
            dec.conv = [list(cs) for cs in conv_bak]
        if pos_bak is not None and ring_bak is not None and dec._kv_ring is not None:
            dec._kv_ring[:, :, :, pos_bak, :] = ring_bak

    def _sync_exact_cache():
        if cache_state is None:
            return
        try:
            for item in reversed(pa_queue):
                rb = item[8]
                if rb is not None:
                    _restore_kv_slices(*rb)
            pa_queue.clear()
            if len(committed_tokens) > len(yielded_tokens) and last_popped_rb is not None:
                cpu_pre, conv_bak, pos_bak, ring_bak = last_popped_rb
                n_com, off_r, idx_r, off_f = cpu_pre
                _restore_kv_slices(conv_bak, pos_bak, ring_bak)
                for v in dec._ring_visits:
                    kv = dec.caches[0][v][0]
                    kv.offset = off_r
                    kv._idx = idx_r
                for v in dec._full_visits:
                    dec.caches[0][v][0].offset = off_f
                del committed_tokens[n_com:]
            dec.flush()
            mx.eval([c.state for c in cache])
            tail = yielded_tokens[len(committed_tokens):]
            if tail:
                model(mx.array([tail], mx.int32), cache=cache)
                mx.eval([c.state for c in cache])
            cache_state["cached_tokens"] = list(yielded_tokens)
        except Exception:
            logging.exception("dflash2 _sync_exact_cache failed")

    try:
        y_tok = int(y.item())
        stats["cycles"] += 1
        stats["tokens"] += 1
        yielded_tokens.append(y_tok)
        yield y_tok, None, False
        n = 1
        if n >= max_tokens:
            return

        anchor = y
        anchor_int = y_tok
        W_cfg = int(os.environ.get("LOOPLING_DFLASH2_W", "6"))
        use_budget = os.environ.get("LOOPLING_DFLASH2_BUDGET", "1") not in ("0", "", "false", "False")
        use_lookup = os.environ.get("LOOPLING_DFLASH2_LOOKUP", "1") not in ("0", "", "false", "False")
        tree_env = os.environ.get("LOOPLING_DFLASH2_TREE", "")
        use_tree = (tree_env not in ("0", "")) if tree_env else use_budget
        tap_check = tap_set | {"F"}

        budget = getattr(dr, "_budget", None) if use_budget else None
        if use_budget and budget is None:
            budget = dr._budget = VerifyBudget()
        costs = getattr(dr, "_costs", None) if use_budget else None
        if use_budget and costs is None:
            costs = dr._costs = MoEStepCosts(dr.qbits)
        elif costs is not None:
            costs.ar_cooldown = 0
            costs.low_streak = 0
            costs.ar_backoff = 4
            costs.ema_ratio = 1.15

        lookup = None
        if use_lookup:
            lookup = LookupIndex()
            p_list = list(full_prompt) if full_prompt is not None else prompt.tolist()
            lookup.append(p_list)
            lookup.append_one(y_tok)

        deferred_feats: list[mx.array] = []

        def flush_deferred() -> None:
            if deferred_feats:
                f_cat = deferred_feats[0] if len(deferred_feats) == 1 else mx.concatenate(deferred_feats, axis=1)
                dr.append_ctx(f_cat)
                deferred_feats.clear()
                mx.async_eval(dr.ck, dr.cv)

        is_greedy = sampler is None or getattr(sampler, "_sampler_key", None) == ("greedy",)

        def ar_step(cur_anchor: mx.array, cur_anchor_int: int):
            tar0 = time.perf_counter()
            y_cur, tp = dec.step_with_taps(cur_anchor, tap_visits=tap_set, sampler=sampler)
            committed_tokens.append(cur_anchor_int)
            f1 = _feats_from_dict(tp, tap_keys)
            mx.eval(y_cur)
            mx.async_eval(*dec.kv_eval_targets(), dec.conv, f1)
            deferred_feats.append(f1)
            if costs is not None:
                costs.observe_ar((time.perf_counter() - tar0) * 1000.0)
            return y_cur, int(y_cur.item())

        prefetched_fwd = None
        prefetched_npos = 0
        prefetched_t0 = 0.0

        use_pipe_ahead = (
            os.environ.get("LOOPLING_DFLASH2_PIPE_AHEAD", "1") not in ("0", "", "false", "False")
            and dr._fused
            and W_cfg >= 2
            and os.environ.get("LOOPLING_COMPILE_VERIFY", "1") not in ("0", "")
            and not (tm.loopcd.enabled_for(passes) and tm.loopcd.mode != "hidden")
        )
        off_cpu = int(dec.caches[0][0][0].offset)
        if use_pipe_ahead and max_tokens > 1:
            if not dec.ensure_gpu_cap(max_tokens + 16):
                use_pipe_ahead = False
            else:
                dr._ensure_cap(dr.n + max_tokens + 32)
        tau_cap = off_cpu + max_tokens + 32

        pa_depth = max(1, min(2, int(os.environ.get("LOOPLING_DFLASH2_PA", "2"))))
        std_samp = None if is_greedy else _std_sampler_params(sampler)
        use_coupled = (
            use_pipe_ahead
            and (std_samp is not None)
            and os.environ.get("LOOPLING_DFLASH2_COUPLED_SAMPLE", "1") not in ("0", "", "false", "False")
        )
        if use_coupled:
            _dt_env = os.environ.get("LOOPLING_DFLASH2_DRAFT_TEMP", "")
            draft_temp = float(_dt_env) if _dt_env else min(std_samp[0], 0.25)
            pa_tree = False
        else:
            draft_temp = 0.0
            pa_tree = bool(use_tree or use_budget or not is_greedy)
        fscale = (12 + 12 * passes) / 24.0
        pa_costs_env = os.environ.get("LOOPLING_DFLASH2_PA_COSTS")
        if pa_costs_env:
            _pa_c_ov = mx.array([float(x) for x in pa_costs_env.split(",")], mx.float32)
            pa_cost_map = {4: _pa_c_ov, 6: _pa_c_ov, 8: _pa_c_ov}
        else:
            # Proven #85 empirical verify_w_gpu(W_max, w_act) + fused draft marginal cost curves on p3.
            s = fscale / 2.0
            _c_def = mx.array([c * s for c in (0.0, 9.6, 17.8, 18.7, 19.6, 20.5, 21.3, 22.1, 22.8)], mx.float32)
            pa_cost_map = {4: _c_def, 6: _c_def, 8: _c_def}
        pa_priors_env = os.environ.get("LOOPLING_DFLASH2_PA_PRIORS")
        if pa_priors_env:
            pa_prior_cfg = mx.array([float(x) for x in pa_priors_env.split(",")], mx.float32)
        else:
            pa_prior_cfg = mx.array([VerifyBudget.PRIOR_SLOPE, *VerifyBudget.PRIOR_INTERCEPTS], mx.float32)

        pa_adapt = bool(use_budget) and os.environ.get("LOOPLING_DFLASH2_PA_ADAPT", "1") not in ("0", "", "false", "False")
        pa_w_run = min(W_cfg, 6) if (pa_adapt and W_cfg > 6) else W_cfg
        pa_ema_tpc = 3.0
        pa_ema_ms = 21.5 * (fscale / 2.0)
        pa_t_ar = 9.6 * (fscale / 2.0)
        pa_low_streak = 0
        pa_ar_burst = 0
        pa_ar_backoff = 16
        pa_packed = None
        pa_packed_w = 0
        pa_last_t = time.perf_counter()

        def _pa_cost_for(w_val: int) -> mx.array:
            if w_val <= 4:
                return pa_cost_map[4]
            if w_val <= 6:
                return pa_cost_map[6]
            return pa_cost_map[8]

        def _draft_pa(anc_tok, w_val: int, p_arr):
            if use_coupled:
                u_d = mx.random.uniform(shape=(w_val - 1,), dtype=mx.float32)
                return dr.draft_and_pack_coupled_gpu(anc_tok, w_val, u_d, draft_temp, pos_arr=p_arr)
            return dr.draft_and_pack_gpu(
                anc_tok, w_val, pos_arr=p_arr, use_tree=pa_tree, cost_arr=_pa_cost_for(w_val), prior_cfg=pa_prior_cfg
            )

        def _launch_pa_cycle(packed, meta_r, meta_f, dr_pos, binfo, ring_may_wrap, launch_next_draft=True, next_w=None):
            if use_coupled:
                blk_c, par_c, dep_c, ca_c, tm_c, q_ids, q_probs, q_child = packed
                wa_c = None
                w_c = int(blk_c.shape[0])
                u_acc = mx.random.uniform(shape=(w_c - 1,), dtype=mx.float32)
                coupled_arg = (q_ids, q_probs, q_child, u_acc)
            elif pa_tree:
                blk_c, par_c, dep_c, ca_c, tm_c, wa_c = packed
                w_c = int(blk_c.shape[0])
                coupled_arg = None
            else:
                blk_c, par_c, dep_c, ca_c, tm_c = packed
                wa_c = None
                w_c = int(blk_c.shape[0])
                coupled_arg = None
            walk_res, kv_news, conv_ws, tap_buf = dec.verify_w_gpu(
                blk_c,
                meta_r=meta_r,
                meta_f=meta_f,
                depths_arr=dep_c,
                conv_anc=ca_c,
                tree_meta=tm_c,
                max_depth=w_c - 1,
                binfo=binfo,
                tau_cap=tau_cap,
                tap_visits=tap_check,
                par_arr=par_c,
                dr_pos=dr_pos,
                use_stage=False,
                w_act=wa_c,
                sampler=sampler,
                coupled=coupled_arg,
            )
            ovf_c = None
            if use_coupled:
                walk_res, ovf_c = walk_res
            (
                path_pad, emitted, keep_arr, last_slot, anc_nxt,
                slot_r_old, end_f_old, dr_pos_old, meta_r_nxt, meta_f_nxt, dr_pos_nxt,
            ) = walk_res
            rb = _snap_rb(slot_r_old, w_c, need_ring=ring_may_wrap) if (cache_state is not None or use_coupled) else None
            eval_a = [
                emitted, keep_arr, path_pad, last_slot, anc_nxt,
                slot_r_old, end_f_old, dr_pos_old, meta_r_nxt, meta_f_nxt, dr_pos_nxt,
                *kv_news,
            ]
            if ovf_c is not None:
                eval_a.append(ovf_c)
            if rb is not None and rb[2] is not None:
                eval_a.append(rb[2])
            for cw in conv_ws:
                eval_a.extend(cw)
            eval_a.extend(tap_buf.values())
            mx.async_eval(*eval_a)

            dec.commit_w_gpu(
                kv_news, conv_ws, path_pad, keep_arr, last_slot, slot_r_old, end_f_old, w_c,
                ring_may_wrap=ring_may_wrap,
            )
            f_w = _feats_from_dict(tap_buf, tap_keys)
            f_pad = f_w[:, path_pad]
            dr.append_ctx_gpu(f_pad, dr_pos_old)

            eval_b = [*dec.kv_eval_targets()]
            for cs in dec.conv:
                eval_b.extend(cs)
            eval_b.extend(dr.ck)
            eval_b.extend(dr.cv)
            if launch_next_draft:
                wn = w_c if next_w is None else int(next_w)
                packed_nxt = _draft_pa(anc_nxt, wn, dr_pos_nxt)
                eval_b.extend(packed_nxt)
            else:
                packed_nxt = None
            mx.async_eval(*eval_b)
            return (emitted, keep_arr, anc_nxt, meta_r_nxt, meta_f_nxt, dr_pos_nxt, packed_nxt, w_c, rb, ovf_c)

        def _can_launch_pa(w_launch: int):
            hi_d = sum(item[7] for item in pa_queue)
            lo_d = len(pa_queue)
            if n + hi_d + w_launch > max_tokens or not dec._kv_stacked or dr.ck[0] is None:
                return False, None
            ok_b, binfo = dec._cw_safe_bounds_range(off_cpu + lo_d, off_cpu + hi_d, w_launch - 1)
            if not ok_b:
                return False, None
            cap_r, cap_f, max_size_r = binfo[4], binfo[5], binfo[6]
            if (
                off_cpu + hi_d + w_launch > cap_f
                or (cap_r < max_size_r and off_cpu + hi_d + w_launch > cap_r)
                or dr.n + hi_d + w_launch + dr.block > int(dr.ck[0].shape[2])
            ):
                return False, None
            return True, binfo

        while n < max_tokens:
            if use_pipe_ahead:
                l_len_pa = lookup.match_len if lookup is not None else 0
                want_lookup_bypass = (
                    lookup is not None and _LOOKUP_BYPASS_MIN > 0 and l_len_pa >= _LOOKUP_BYPASS_MIN
                )
                want_ar_burst = pa_ar_burst > 0 and not want_lookup_bypass
                if not want_lookup_bypass and not want_ar_burst:
                    if deferred_feats:
                        flush_deferred()
                        if not dec._kv_stacked:
                            dec.ensure_gpu_cap(max_tokens - n + 16)
                            off_cpu = int(dec.caches[0][0][0].offset)
                        dr._ensure_cap(dr.n + max_tokens - n + 32)
                    while len(pa_queue) < pa_depth:
                        qlen = len(pa_queue)
                        w_l = pa_w_run
                        ok_pa, binfo_pa = _can_launch_pa(w_l)
                        if not ok_pa:
                            break
                        if qlen == 0:
                            if pa_packed is None or pa_packed_w != w_l:
                                pa_packed = _draft_pa(anchor, w_l, FD._wpos(dr.n))
                                pa_packed_w = w_l
                            meta_r0, meta_f0 = dec.get_gpu_meta(staged=False)
                            dr_pos0 = FD._wpos(dr.n)
                            lnxt = (pa_depth >= 2) and (n + 2 * w_l <= max_tokens)
                            pa_last_t = time.perf_counter()
                            pa_queue.append(
                                _launch_pa_cycle(
                                    pa_packed, meta_r0, meta_f0, dr_pos0, binfo_pa,
                                    ring_may_wrap=(off_cpu + w_l > binfo_pa[6]),
                                    launch_next_draft=lnxt,
                                    next_w=w_l,
                                )
                            )
                            pa_packed = None
                            pa_packed_w = 0
                        else:
                            prev = pa_queue[-1]
                            p_nxt = prev[6]
                            if p_nxt is None or int(p_nxt[0].shape[0]) != w_l:
                                break
                            hi_d = sum(item[7] for item in pa_queue)
                            lnxt = (n + hi_d + 2 * w_l <= max_tokens)
                            pa_queue.append(
                                _launch_pa_cycle(
                                    p_nxt, prev[3], prev[4], prev[5], binfo_pa,
                                    ring_may_wrap=(off_cpu + hi_d + w_l > binfo_pa[6]),
                                    launch_next_draft=lnxt,
                                    next_w=w_l,
                                )
                            )

                if pa_queue:
                    emitted_c, keep_arr_c, anc_nxt, _, _, _, p_nxt, w_done, rb_c, ovf_c = pa_queue.popleft()
                    if ovf_c is not None:
                        mx.eval(emitted_c, keep_arr_c, ovf_c)
                        if int(mx.max(ovf_c).item()) != 0:
                            for item in reversed(pa_queue):
                                if item[8] is not None:
                                    _restore_kv_slices(*item[8])
                            pa_queue.clear()
                            if rb_c is not None:
                                _restore_kv_slices(*rb_c)
                            pa_packed = None
                            pa_packed_w = 0
                            use_coupled = False
                            pa_tree = True
                            anchor, tok = ar_step(anchor, anchor_int)
                            anchor_int = tok
                            off_cpu = int(dec.caches[0][0][0].offset)
                            if lookup is not None:
                                lookup.append_one(tok)
                            stats["cycles"] += 1
                            stats["tokens"] += 1
                            yielded_tokens.append(tok)
                            yield tok, None, False
                            n += 1
                            continue
                    else:
                        mx.eval(emitted_c, keep_arr_c)
                    now_t = time.perf_counter()
                    cyc_ms = (now_t - pa_last_t) * 1000.0
                    stats["verify_ms"] += cyc_ms
                    pa_last_t = now_t
                    keep = int(keep_arr_c.item())
                    em = emitted_c.tolist()[:keep]
                    if cache_state is not None:
                        last_popped_rb = (_read_cpu_kv(), *rb_c)
                    committed_tokens.append(anchor_int)
                    if keep > 1:
                        committed_tokens.extend(em[: keep - 1])
                    dec.sync_kv_offsets(keep, staged=False)
                    dr.n += keep
                    off_cpu += keep
                    stats["cycles"] += 1
                    stats["accepted"] += keep - 1
                    if lookup is not None:
                        for tok_e in em:
                            lookup.append_one(tok_e)
                    if pa_adapt:
                        pa_ema_tpc = 0.72 * pa_ema_tpc + 0.28 * keep
                        if 4.0 <= cyc_ms <= 60.0:
                            pa_ema_ms = 0.75 * pa_ema_ms + 0.25 * cyc_ms
                        # Keep W=6 as neural floor (proved 134.13 MEAN8 in #85); promote 6->8 only on high-tpc prompts.
                        if W_cfg >= 8 and pa_w_run < 8 and pa_ema_tpc >= 4.4:
                            pa_w_run = 8
                        elif pa_w_run == 8 and pa_ema_tpc < 3.8:
                            pa_w_run = 6
                        # AR fallback only on genuinely low-acceptance prompts (e.g. P1) where spec trails AR.
                        spec_rate = pa_ema_tpc / max(1.0, pa_ema_ms)
                        ar_rate = 1.0 / max(1.0, pa_t_ar)
                        if keep <= 2 and pa_ema_tpc < 2.05 and spec_rate < 0.92 * ar_rate:
                            pa_low_streak += 1
                            if pa_low_streak >= 4:
                                pa_ar_burst = pa_ar_backoff
                                pa_ar_backoff = min(32, pa_ar_backoff + 8)
                                pa_low_streak = 0
                        elif spec_rate >= 1.00 * ar_rate:
                            pa_low_streak = 0
                            if keep >= 3:
                                pa_ar_backoff = 16
                    anchor_int = em[keep - 1]
                    if not pa_queue:
                        anchor = anc_nxt
                        pa_packed = p_nxt
                        pa_packed_w = int(p_nxt[0].shape[0]) if p_nxt is not None else 0
                    for j in range(keep):
                        stats["tokens"] += 1
                        yielded_tokens.append(em[j])
                        yield em[j], None, (j < keep - 1)
                        n += 1
                        if n >= max_tokens:
                            break
                    continue
                pa_packed = None
                pa_packed_w = 0
                if want_ar_burst:
                    pa_ar_burst -= 1
                    tar0 = time.perf_counter()
                    anchor, tok = ar_step(anchor, anchor_int)
                    anchor_int = tok
                    pa_t_ar = 0.75 * pa_t_ar + 0.25 * ((time.perf_counter() - tar0) * 1000.0)
                    off_cpu = int(dec.caches[0][0][0].offset)
                    if lookup is not None:
                        lookup.append_one(tok)
                    if pa_ar_burst == 0:
                        flush_deferred()
                        if not dec._kv_stacked:
                            dec.ensure_gpu_cap(max_tokens - n + 16)
                            off_cpu = int(dec.caches[0][0][0].offset)
                        dr._ensure_cap(dr.n + max_tokens - n + 32)
                        pa_ema_tpc = 2.6
                        pa_w_run = min(6, W_cfg)
                    stats["cycles"] += 1
                    stats["tokens"] += 1
                    yielded_tokens.append(tok)
                    yield tok, None, False
                    n += 1
                    continue

            rem = max_tokens - n
            w_cur = max(1, min(W_cfg, dr.block, rem))
            l_len = lookup.match_len if lookup is not None else 0
            w_lkp = max(w_cur, min(_LOOKUP_W, rem)) if (_LOOKUP_BYPASS_MIN > 0 and l_len >= _LOOKUP_BYPASS_MIN) else w_cur
            l_props = lookup.proposals(w_lkp - 1) if (lookup is not None and l_len >= LookupIndex.MIN_MATCH) else []

            in_cooldown = costs is not None and costs.ar_cooldown > 0 and l_len < 4
            if w_cur == 1 or in_cooldown:
                prefetched_fwd = None
                if costs is not None and costs.ar_cooldown > 0:
                    costs.ar_cooldown -= 1
                anchor, tok = ar_step(anchor, anchor_int)
                anchor_int = tok
                off_cpu = int(dec.caches[0][0][0].offset)
                if lookup is not None:
                    lookup.append_one(tok)
                stats["cycles"] += 1
                stats["tokens"] += 1
                yielded_tokens.append(tok)
                yield tok, None, False
                n += 1
                continue

            did_lookup_bypass = bool(l_props and _LOOKUP_BYPASS_MIN > 0 and l_len >= _LOOKUP_BYPASS_MIN)
            if not did_lookup_bypass:
                flush_deferred()
            pf = prefetched_fwd if (prefetched_fwd is not None and prefetched_npos == w_cur - 1) else None
            td0 = prefetched_t0 if pf is not None else time.perf_counter()
            prefetched_fwd = None
            if did_lookup_bypass:
                anc_int = int(anchor.item())
                toks_lp = [anc_int] + l_props[: w_lkp - 1]
                blk = mx.array(toks_lp, mx.int32)
                parents = tuple(range(-1, len(toks_lp) - 1))
                depths = tuple(range(len(toks_lp)))
                raw_probs = [-1.0] * len(toks_lp)
                td_ms = (time.perf_counter() - td0) * 1000.0
            else:
                if use_budget or use_tree or l_props:
                    blk, parents, depths, raw_probs = dr.draft_hybrid_tree(
                        anchor,
                        w_max=w_cur,
                        budget=budget,
                        step_costs=costs,
                        lookup_props=l_props,
                        lookup_len=l_len,
                        use_tree=use_tree,
                        prefetched=pf,
                    )
                else:
                    blk = dr.draft_chain(anchor, width=w_cur, prefetched=pf)
                    parents = tuple(range(-1, w_cur - 1))
                    depths = tuple(range(w_cur))
                    raw_probs = [-1.0] * w_cur
                td_ms = (time.perf_counter() - td0) * 1000.0
                if costs is not None:
                    costs.observe_draft(td_ms)
            if blk is None:
                stats["draft_ms"] += td_ms
                if costs is not None:
                    costs.ar_cooldown = max(costs.ar_cooldown, 3)
                anchor, tok = ar_step(anchor, anchor_int)
                anchor_int = tok
                off_cpu = int(dec.caches[0][0][0].offset)
                if lookup is not None:
                    lookup.append_one(tok)
                stats["cycles"] += 1
                stats["tokens"] += 1
                yielded_tokens.append(tok)
                yield tok, None, False
                n += 1
                continue
            stats["draft_ms"] += td_ms

            w_act = len(parents)
            is_tree = parents != tuple(range(-1, w_act - 1))
            if is_tree:
                _, ca, tm_tbl = build_tree_tables(parents, depths)
            else:
                ca = tm_tbl = None

            tv0 = time.perf_counter()
            if is_tree:
                t, kv_news, conv_ws, tap_buf = dec.verify_w(
                    blk, tap_visits=tap_check, parents=parents, depths=depths, conv_anc=ca, tree_meta=tm_tbl,
                    sampler=sampler,
                )
            else:
                t, kv_news, conv_ws, tap_buf = dec.verify_w(blk, tap_visits=tap_check, sampler=sampler)
            mx.eval(t)
            tv_ms = (time.perf_counter() - tv0) * 1000.0

            tl = t.tolist()
            bl = blk.tolist()
            if budget is not None:
                budget.observe(parents, depths, raw_probs, bl, tl)

            children = [[] for _ in range(w_act)]
            for s in range(1, w_act):
                children[parents[s]].append((s, bl[s]))

            path = [0]
            cur_slot = 0
            while True:
                target_tok = tl[cur_slot]
                matched = None
                for c_slot, c_tok in children[cur_slot]:
                    if c_tok == target_tok:
                        matched = c_slot
                        break
                if matched is None:
                    bonus_tok = target_tok
                    break
                path.append(matched)
                cur_slot = matched

            keep = len(path)
            stats["cycles"] += 1
            stats["accepted"] += keep - 1

            if lookup is not None:
                for j in range(1, keep):
                    lookup.append_one(bl[path[j]])
                lookup.append_one(bonus_tok)

            tc_ms = (time.perf_counter() - tv0) * 1000.0
            stats["verify_ms"] += tv_ms
            if costs is not None:
                costs.observe_verify(w_act, tc_ms, keep, td_ms)

            if n + keep < max_tokens:
                anchor_next = t[cur_slot : cur_slot + 1]
                if cache_state is not None:
                    has_ring = bool(dec._kv_stacked and dec._ring_visits)
                    if has_ring:
                        kv_r0 = dec.caches[0][dec._ring_visits[0]][0]
                        max_size_r = int(kv_r0.max_size)
                        slot_r0 = 0 if int(kv_r0._idx) == max_size_r else int(kv_r0._idx)
                        need_ring = (off_cpu + keep) > max_size_r
                    else:
                        slot_r0 = 0
                        need_ring = False
                    rb_np = _snap_rb(slot_r0, keep, need_ring=need_ring)
                    if rb_np[2] is not None:
                        mx.async_eval(rb_np[2])
                    last_popped_rb = (_read_cpu_kv(), *rb_np)
                dec.commit_w(kv_news, conv_ws, keep, path=path)
                committed_tokens.extend(bl[path[j]] for j in range(keep))
                off_cpu = int(dec.caches[0][0][0].offset)
                f_w = _feats_from_dict(tap_buf, tap_keys)
                f_keep = f_w[:, :keep] if path == list(range(keep)) else f_w[:, mx.array(path, mx.int32)]
                anchor = anchor_next

                rem_next = max_tokens - (n + keep)
                w_next = max(1, min(W_cfg, dr.block, rem_next))
                l_next = lookup.match_len if lookup is not None else 0
                byp_next = _LOOKUP_BYPASS_MIN > 0 and l_next >= _LOOKUP_BYPASS_MIN
                if did_lookup_bypass and byp_next:
                    # Defer drafter KV update while continuing zero-draft lookup streak (-0.5ms/cycle on P8!).
                    deferred_feats.append(f_keep)
                    mx.async_eval(*dec.kv_eval_targets(), dec.conv, f_keep)
                else:
                    if deferred_feats:
                        deferred_feats.append(f_keep)
                        flush_deferred()
                    else:
                        dr.append_ctx(f_keep)
                    cd_next = costs is not None and costs.ar_cooldown > 0 and l_next < 4
                    if not use_pipe_ahead and w_next >= 2 and not cd_next and not byp_next:
                        prefetched_t0 = time.perf_counter()
                        prefetched_npos = w_next - 1
                        prefetched_fwd = dr._forward_block(anchor, n_pos=prefetched_npos)
                        mx.async_eval(*dec.kv_eval_targets(), dec.conv, dr.ck, dr.cv, *prefetched_fwd)
                    else:
                        mx.async_eval(*dec.kv_eval_targets(), dec.conv, dr.ck, dr.cv)

            anchor_int = bonus_tok
            for j in range(1, keep):
                tok_d = bl[path[j]]
                stats["tokens"] += 1
                yielded_tokens.append(tok_d)
                yield tok_d, None, True
                n += 1
                if n >= max_tokens:
                    break
            if n < max_tokens:
                stats["tokens"] += 1
                yielded_tokens.append(bonus_tok)
                yield bonus_tok, None, False
                n += 1
    finally:
        _sync_exact_cache()
