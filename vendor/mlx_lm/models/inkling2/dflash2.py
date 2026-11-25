"""DFlash 2 block-diffusion speculative drafter for Inkling2 / Loopling (opt-in LOOPLING_MTP=dflash2).

Ports Splash's `incoai/Inkling-2-Mini-DFlash2` model (5-layer non-causal Qwen3 backbone with 2-tap dynamic convolutions
and low-rank bilinear candidate selector) paired with BatchDecoder's bit-exact causal verify engine.
"""
from __future__ import annotations

import heapq
import json
import math
import os
import time
from collections import deque

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.inkling2 import fastdecode as FD
from mlx_lm.models.inkling2.rows import qmv as RQ

DFLASH2_DIR = os.environ.get(
    "LOOPLING_DFLASH2_DIR",
    os.path.expanduser("~/loopling/models/Inkling-2-Mini-DFlash2"),
)
_EPS = 1e-6


def _quantize_if(w: mx.array, bits: int):
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
    succ_vecs = succ_cb[top_idx].astype(mx.float32)  # (P, 16, R)

    anc = anchor_tok.reshape(())
    pred0 = pred_cb[anc].astype(mx.float32) * sel_h[0]  # (R,)
    edge0 = succ_vecs[0] @ pred0  # (16,)
    sum0 = unary[0] + edge0
    sc0 = softcap * mx.tanh(sum0 / softcap) if softcap > 0.0 else sum0

    # Conditional edges for positions 1..P-1 across all 16 predecessors x 16 successors
    pred_vecs = pred_cb[top_idx[:-1]].astype(mx.float32) * sel_h[1:, None, :]  # (P-1, 16, R)
    edges = mx.matmul(pred_vecs, succ_vecs[1:].swapaxes(-1, -2))  # (P-1, 16, 16)
    sums = unary[1:, None, :] + edges
    sc16 = softcap * mx.tanh(sums / softcap) if softcap > 0.0 else sums
    return top_idx, sc0, sc16


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

    PRIOR_SLOPE = 0.85
    PRIOR_INTERCEPTS = (0.20, 0.0, -0.10, -0.15, -0.20, -0.25, -0.25)
    SLOPE_PREC = 10.0
    INTERCEPT_PREC = 2.0
    MAX_LABELS = 4096
    REFIT_CYCLES = 16

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
        self.t_ar = 10.2
        self.t_draft = 2.8 if quant_bits else 5.5
        self.t_v2 = 17.5
        self.t_v8 = 23.8
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
            self.t_v2 = 1.72 * ms
            self.t_v8 = 2.35 * ms
        else:
            self.t_ar = 0.8 * self.t_ar + 0.2 * ms

    def observe_draft(self, ms: float) -> None:
        if ms > 0.0:
            self.t_draft = 0.75 * self.t_draft + 0.25 * ms

    def observe_verify(self, width: int, ms: float, keep: int, draft_ms: float) -> None:
        if ms <= 0.0 or width < 2:
            return
        if not self._seeded_verify:
            self._seeded_verify = True
            if width >= 6:
                self.t_v8 = ms
                self.t_v2 = max(self.t_ar * 1.35, ms * 0.73)
            else:
                self.t_v2 = ms
                self.t_v8 = ms * 1.35
        else:
            u = max(0.0, min(1.0, (width - 2) / 6.0))
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
            self._lm_q4 = (wq4, sc4, bi4)
            to_eval.extend([wq4, sc4, bi4])
        self._df_pre_rope, self._df_post_attn = _get_df_compiled(
            self.nh, self.nkv, self.hd, self.inter, self.qbits
        )
        to_eval.extend([self._mask_emb, self._mask_7])
        mx.eval(*to_eval)

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

    def reset(self) -> None:
        self.ck = [None] * self.nl
        self.cv = [None] * self.nl
        self.n = 0

    def _ensure_cap(self, need: int) -> None:
        cur = 0 if self.ck[0] is None else int(self.ck[0].shape[2])
        if need <= cur:
            return
        new_cap = max(512, ((need + 511) // 512) * 512)
        for i in range(self.nl):
            nk = mx.zeros((1, self.nkv, new_cap, self.hd), mx.bfloat16)
            nv = mx.zeros((1, self.nkv, new_cap, self.hd), mx.bfloat16)
            if self.ck[i] is not None and self.n > 0:
                nk[:, :, : self.n, :] = self.ck[i][:, :, : self.n, :]
                nv[:, :, : self.n, :] = self.cv[i][:, :, : self.n, :]
            self.ck[i], self.cv[i] = nk, nv

    def append_ctx(self, feats: mx.array) -> None:
        """Project target taps `(1, N, 4, H)` -> context K/V for all 5 draft layers."""
        n_tok = int(feats.shape[1])
        if n_tok == 0:
            return
        self._ensure_cap(self.n + n_tok)
        flat = feats.reshape(n_tok, 4 * self.H)
        ctx = _rms(self._lin(flat, self.fc), self.hidden_norm)  # (N, H)
        pos = self.n
        for i, lw in enumerate(self.layers):
            kv = self._lin(ctx, lw["kv"])  # (N, 2*nkv*hd)
            kw = self.nkv * self.hd
            k = kv[:, :kw].reshape(1, n_tok, self.nkv, self.hd)
            v = kv[:, kw:].reshape(1, n_tok, self.nkv, self.hd)
            k = _rms(k, lw["k_norm"]).transpose(0, 2, 1, 3)
            k = mx.fast.rope(k, self.hd, traditional=False, base=self.theta, scale=1.0, offset=pos)
            v = v.transpose(0, 2, 1, 3)
            wpos = FD._wpos(pos)
            ck_i = self.ck[i]
            self.ck[i] = None
            self.ck[i] = mx.slice_update(ck_i, k, wpos, axes=(2,))
            cv_i = self.cv[i]
            self.cv[i] = None
            self.cv[i] = mx.slice_update(cv_i, v, wpos, axes=(2,))
        self.n += n_tok

    def _ctx_slice(self, i: int):
        if self.n == 0:
            return None, None
        start = max(0, self.n - self.window)
        return self.ck[i][:, :, start : self.n, :], self.cv[i][:, :, start : self.n, :]

    def _forward_block(self, anchor_tok: mx.array, n_pos: int = 7):
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
            raw_logits = mx.quantized_matmul(
                hid, wq4, scales=sc4, biases=bi4, transpose=True, group_size=64, bits=4, mode="affine"
            )
        elif self._lm_aff8:
            raw_logits = RQ.qmv_rows(hid, *self._lm_t)[:, : self._unpadded_vocab]
        else:
            raw_logits = self.target.lm_head(hid)[:, : self._unpadded_vocab]
        sel_h = self._lin(hid, self.sel_proj).astype(mx.float32)  # (n_pos, R)
        return _selector_scores(raw_logits, sel_h, self.pred_cb, self.succ_cb, anchor_tok, self.softcap)

    def draft_chain(self, anchor_tok: mx.array, width: int = 8) -> mx.array:
        """Return `[anchor, d_1 .. d_{width-1}]` `(width,)` int32 via greedy selector chain."""
        width = max(2, min(self.block, int(width)))
        top_idx, sc0, sc16 = self._forward_block(anchor_tok, n_pos=width - 1)
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
    ):
        """Return `(toks (W,) i32, parents, depths, raw_probs)` or `(None, ...)` when StepCosts picks W=1."""
        w_max = max(2, min(self.block, int(w_max)))
        top_idx, sc0, sc16 = self._forward_block(anchor_tok, n_pos=w_max - 1)
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


_DF2_INST: dict = {}


def _get_drafter(model, draft_dir: str = DFLASH2_DIR) -> DFlash2Drafter:
    lm_w = model.lm_head["weight"]
    qbits = int(os.environ.get("LOOPLING_DFLASH2_QUANT", "4"))
    key = (id(model), draft_dir, qbits)
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
    **kw,
):
    """Bit-exact causal-verify speculative decoding with Splash DFlash 2 drafter."""
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.models.inkling2.batchdecode import BatchDecoder, build_tree_tables
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
    if n_p > 1:
        tm.tap_visits = tap_set
        while n_p - pos > 1:
            tm.tap_buf = {}
            m = min(prefill_step_size, n_p - 1 - pos)
            _ = model(prompt[pos : pos + m][None], cache=cache)
            f = feats_from_buf(tm, taps)
            dr.append_ctx(f)
            mx.eval([c.state for c in cache], dr.ck, dr.cv)
            pos += m
            mx.clear_cache()
        tm.tap_visits = None
        tm.tap_buf = {}

    dec = BatchDecoder(model, [cache])
    if not dec._b1_fast:
        raise RuntimeError("dflash2_exact_generate_step requires BatchDecoder._b1_fast")

    y, tap0 = dec.step_with_taps(prompt[pos:], tap_visits=tap_set)
    f0 = _feats_from_dict(tap0, tap_keys)
    dr.append_ctx(f0)
    mx.eval(y, dr.ck, dr.cv, *dec.kv_eval_targets(), dec.conv)
    y_tok = int(y.item())
    stats["cycles"] += 1
    stats["tokens"] += 1
    yield y_tok, None, False
    n = 1
    if n >= max_tokens:
        return

    anchor = y
    W_cfg = int(os.environ.get("LOOPLING_DFLASH2_W", "8"))
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
        p_list = prompt.tolist()
        lookup.append(p_list)
        lookup.append_one(y_tok)

    deferred_feats: list[mx.array] = []

    def flush_deferred() -> None:
        if deferred_feats:
            f_cat = deferred_feats[0] if len(deferred_feats) == 1 else mx.concatenate(deferred_feats, axis=1)
            dr.append_ctx(f_cat)
            deferred_feats.clear()
            mx.async_eval(dr.ck, dr.cv)

    def ar_step(cur_anchor: mx.array):
        tar0 = time.perf_counter()
        y_cur, tp = dec.step_with_taps(cur_anchor, tap_visits=tap_set)
        f1 = _feats_from_dict(tp, tap_keys)
        mx.eval(y_cur)
        mx.async_eval(*dec.kv_eval_targets(), dec.conv, f1)
        deferred_feats.append(f1)
        if costs is not None:
            costs.observe_ar((time.perf_counter() - tar0) * 1000.0)
        return y_cur, int(y_cur.item())

    while n < max_tokens:
        rem = max_tokens - n
        w_cur = max(1, min(W_cfg, dr.block, rem))
        l_len = lookup.match_len if lookup is not None else 0
        l_props = lookup.proposals(w_cur - 1) if (lookup is not None and l_len >= LookupIndex.MIN_MATCH) else []

        in_cooldown = costs is not None and costs.ar_cooldown > 0 and l_len < 4
        if w_cur == 1 or in_cooldown:
            if costs is not None and costs.ar_cooldown > 0:
                costs.ar_cooldown -= 1
            anchor, tok = ar_step(anchor)
            if lookup is not None:
                lookup.append_one(tok)
            stats["cycles"] += 1
            stats["tokens"] += 1
            yield tok, None, False
            n += 1
            continue

        flush_deferred()
        td0 = time.perf_counter()
        if use_budget or use_tree or l_props:
            blk, parents, depths, raw_probs = dr.draft_hybrid_tree(
                anchor,
                w_max=w_cur,
                budget=budget,
                step_costs=costs,
                lookup_props=l_props,
                lookup_len=l_len,
                use_tree=use_tree,
            )
        else:
            blk = dr.draft_chain(anchor, width=w_cur)
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
            anchor, tok = ar_step(anchor)
            if lookup is not None:
                lookup.append_one(tok)
            stats["cycles"] += 1
            stats["tokens"] += 1
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
                blk, tap_visits=tap_check, parents=parents, depths=depths, conv_anc=ca, tree_meta=tm_tbl
            )
        else:
            t, kv_news, conv_ws, tap_buf = dec.verify_w(blk, tap_visits=tap_check)
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

        if n + keep < max_tokens:
            anchor_next = t[cur_slot : cur_slot + 1]
            dec.commit_w(kv_news, conv_ws, keep, path=path)
            f_w = _feats_from_dict(tap_buf, tap_keys)
            f_keep = f_w[:, :keep] if path == list(range(keep)) else f_w[:, mx.array(path, mx.int32)]
            dr.append_ctx(f_keep)
            mx.async_eval(*dec.kv_eval_targets(), dec.conv, dr.ck, dr.cv)
            anchor = anchor_next

        tc_ms = (time.perf_counter() - tv0) * 1000.0
        stats["verify_ms"] += tv_ms
        if costs is not None:
            costs.observe_verify(w_act, tc_ms, keep, td_ms)

        if lookup is not None:
            for j in range(1, keep):
                lookup.append_one(bl[path[j]])
            lookup.append_one(bonus_tok)

        for j in range(1, keep):
            stats["tokens"] += 1
            yield bl[path[j]], None, True
            n += 1
            if n >= max_tokens:
                break
        if n < max_tokens:
            stats["tokens"] += 1
            yield bonus_tok, None, False
            n += 1
