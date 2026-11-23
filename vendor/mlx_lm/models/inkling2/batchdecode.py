"""Batched decode: C independent streams advance one token in ONE forward, and every row is bitwise equal to the B=1
fast decode path (fastdecode.py) of that stream. Exact by construction per op; checked end to end by
bench/batch_dec.py (tokens of every stream == its single-stream greedy tokens).

Which formulation keeps a C-row op bitwise equal to C separate B=1 calls (bench/rows_ops.py, bench/rows_f32.py):
  * elementwise, norms, l2 clips, depth attention, relative logits, router top-k kernel, experts (gather_qmv per row),
    dynamic short conv, softcap, argmax: plain batching is already row invariant.
  * 8-bit QuantizedLinear (attention projections, dynamic-conv weight_proj, lm_head): plain M=C dispatch changes the
    kernel (qmv -> qmv_wide/qmm). Batch the WEIGHTS instead (stride-0 broadcast, B=C, M=1 each): same qmv kernel per row.
  * f32 Linear (router, message-gate forget/message projections): every zero-copy form (stride-0 broadcast, gather_mm,
    addmm, einsum) collapses to an M=C GEMM. Only a materialized (C, N, K) copy gives a batched M=1 gemv per row.
    The copies are made once per (weight, C) and captured by the compiled segments (weights are frozen: call
    `reset()` after changing model weights).
  * attention core (KV cache update + sdpa): per stream (each stream has its own cache length).
  * log-softmax before argmax: per row, with the same all-axes reduction as generate_step.

Usage:
    dec = BatchDecoder(model, caches)      # caches[i] = prompt cache of stream i (prefilled, as generate_step leaves it)
    toks = dec.step(last_tokens)           # (C,) int -> (C,) next greedy tokens; caches advance in place
    dec.flush()                            # write the batched conv states back into the per-stream caches
"""
import os

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.inkling2 import fastdecode as FD
from mlx_lm.models.inkling2 import layers as L
from mlx_lm.models.inkling2.rows import gemv as RG
from mlx_lm.models.inkling2.rows import qmv as RQ
from mlx_lm.models.inkling2.rows import sdpa as RS

_ROWS = False
# exact row kernels (lane E1/E3/E4): weights read once for all rows; each row bitwise = the B=1 op. Default all on.
_USE = set(filter(None, os.environ.get("LOOPLING_BD_ROWS", "qmv,gemv,sdpa").split(",")))
# timing-only toggles (NOT exact): plain M=C dispatch instead of the invariant formulation
_PLAIN = set(filter(None, os.environ.get("LOOPLING_BD_PLAIN", "").split(",")))  # ql,lin,router,head
_mat: dict = {}
_installed = False


def _rows(x):
    return _ROWS and x.ndim == 3 and x.shape[0] > 1 and x.shape[1] == 1


def _matw(w, C):
    """Materialized (C, N, K) copy of a float weight (N, K), cached per (weight, C)."""
    k = (id(w), C)
    m = _mat.get(k)
    if m is None or m[0] is not w:
        c = mx.contiguous(mx.broadcast_to(w[None], (C,) + w.shape))
        mx.eval(c)
        m = _mat[k] = (w, c)
    return m[1]


def install():
    global _installed
    if _installed:
        return
    _installed = True
    ql_call, lin_call, router_call = nn.QuantizedLinear.__call__, nn.Linear.__call__, L.TopKRouter.__call__

    def qlinear(self, x):
        if not _rows(x) or "ql" in _PLAIN:
            return ql_call(self, x)
        C, K = x.shape[0], x.shape[-1]
        if ("qmv" in _USE and self.mode == "affine" and self.bits == 8 and self.group_size == 64
                and x.dtype == mx.bfloat16 and C <= RQ.MAX_B and K % 256 == 0 and K not in (64, 128)
                and "bias" not in self):
            return RQ.qmv_rows(x.reshape(C, K), self["weight"], self["scales"], self["biases"]).reshape(C, 1, -1)
        bc = lambda a: None if a is None else mx.broadcast_to(a[None], (C,) + a.shape)
        y = mx.quantized_matmul(x, bc(self["weight"]), scales=bc(self["scales"]), biases=bc(self.get("biases")),
                                transpose=True, group_size=self.group_size, bits=self.bits, mode=self.mode)
        if "bias" in self:
            y = y + self["bias"]
        return y

    def linear(self, x):
        if not _rows(x) or "lin" in _PLAIN:
            return lin_call(self, x)
        w = self["weight"]
        if "gemv" in _USE and x.dtype == w.dtype and w.dtype in (mx.float32, mx.bfloat16) and "bias" not in self:
            C, K = x.shape[0], x.shape[-1]
            return RG.gemv_rows(x.reshape(C, K), w).reshape(C, 1, -1)
        y = x @ _matw(w, x.shape[0]).transpose(0, 2, 1)
        if "bias" in self:
            y = y + self["bias"]
        return y

    def router(self, h):
        if not _rows(h) or "router" in _PLAIN:
            return router_call(self, h)
        from mlx_lm.kernels.expert_routing import expert_routing

        assert L._use_kernels()
        if "gemv" in _USE:
            C, K = h.shape[0], h.shape[-1]
            logits = RG.gemv_rows(h.astype(mx.float32).reshape(C, K), self.weight).reshape(C, 1, -1)
        else:
            logits = h.astype(mx.float32) @ _matw(self.weight, h.shape[0]).transpose(0, 2, 1)
        return expert_routing(logits, self.e_score_correction_bias, top_k=self.top_k, multiplier=1.0,
                              score_eps=L._ROUTER_SCORE_EPS, route_scale=self.routed_scaling_factor)

    nn.QuantizedLinear.__call__ = qlinear
    nn.Linear.__call__ = linear
    L.TopKRouter.__call__ = router


class _RowsMode:
    def __enter__(self):
        global _ROWS
        self.prev, _ROWS = _ROWS, True

    def __exit__(self, *a):
        global _ROWS
        _ROWS = self.prev


def _wrap(layer, body):
    def run(*a):
        with _RowsMode():
            return body(layer, *a)

    return mx.compile(run) if FD._COMPILE else run


class BatchDecoder:
    def __init__(self, model, caches):
        install()
        self.model = model
        self.tm = model.model.language_model
        self.caches = caches
        self.C = len(caches)
        self.nv = len(caches[0])
        # batched conv states per visit: 4 arrays (C, 3, dims) f32
        self.conv = [[mx.concatenate([c[v][1].cache[j] for c in caches]) for j in range(4)] for v in range(self.nv)]
        self.seg: dict = {}
        self._head = None

    def flush(self):
        for v in range(self.nv):
            for i, c in enumerate(self.caches):
                for j in range(4):
                    c[v][1][j] = self.conv[v][j][i : i + 1]

    def _segs(self, layer, k):
        key = (id(layer), k)
        s = self.seg.get(key)
        if s is None:
            fd = FD._get_fd(layer)
            w = lambda body: _wrap(layer, body)
            s = self.seg[key] = (fd["make_s1"](*k, wrap=w), fd["make_s2"](*k, wrap=w))
        return s

    def _layer(self, layer, v, h, bs, msg):
        sliding = isinstance(layer, L.SlidingDecoderLayer)
        kvs = [c[v][0] for c in self.caches]
        cs = self.conv[v]
        k = (bs is not None, msg is not None)
        s1, s2 = self._segs(layer, k)
        args = [h]
        if bs is not None:
            args.append(bs)
        if msg is not None:
            args.append(msg)
        if not sliding:
            args.append(mx.concatenate([FD._full_tau(layer, FD._end(kv))[None] for kv in kvs]))
        args += [cs[0], cs[1]]
        q, kk, vv, pb, sinks, gate, w0, w1, nr = s1(*args)
        if "sdpa" in _USE and not any(getattr(kv, "_fd_delta", None) is not None for kv in kvs):
            ks, vs, meta = [], [], []
            for i, kv in enumerate(kvs):
                ka, va = kv.update_and_fetch(kk[i : i + 1], vv[i : i + 1])
                ks.append(ka)
                vs.append(va)
                meta.append(RS.row_meta(kv))
            a = layer.self_attn
            attn_out = RS.sdpa_rows(q, ks, vs, pb, sinks, meta, scale=a.scaling, window=a.sliding_window or 0)
        else:
            outs = [FD._core(layer, q[i : i + 1], kk[i : i + 1], vv[i : i + 1], pb[i : i + 1], sinks[i : i + 1], kv)
                    for i, kv in enumerate(kvs)]
            attn_out = mx.concatenate(outs)
        args2 = [h, nr, attn_out, gate, w0, w1, cs[2], cs[3]]
        if bs is not None:
            args2.append(bs)
        if msg is not None:
            args2.append(msg)
        o, bs_new, *states = s2(*args2)
        self.conv[v] = states
        return o, bs_new

    def hidden(self, tokens):
        tm = self.tm
        h = tm.embed(tokens.reshape(-1, 1))
        dtype = h.dtype
        core = tm.args.loop_core
        layers = tm.layers
        self._v = 0

        def run(ls, h, bs, bank=None):
            for i, layer in enumerate(ls):
                msg = None if bank is None else tm.message_input(bank, i)
                h, bs = self._layer(layer, self._v, h, bs, msg)
                self._v += 1
            return h, bs

        prelude = run(layers[: core.start], h, None)
        h, bs = prelude
        for p in range(tm.args.loop_config["num_passes"]):
            bank = None
            if p > 0:
                bank = (bs[prelude[1].shape[0] + 1 :], L.unweighted_rms_norm(h.astype(mx.float32)))
                h, bs = prelude
            h, bs = run(layers[core.start : core.stop], h, bs, bank)
        h, bs = run(layers[core.stop :], h, bs)
        assert self._v == self.nv
        normed = L.unweighted_rms_norm(h.astype(mx.float32))
        h = tm.final_input(bs, normed)[..., 0, :]
        return tm.norm(h).astype(dtype)

    def logits(self, h):
        m = self.model
        if "head" in _PLAIN:
            logits = m.lm_head(h)
        else:
            with _RowsMode():
                logits = m.lm_head(h)
        text = m.args.text
        softcap = text.final_logit_softcapping
        logits = logits.astype(mx.float32)
        logits = mx.tanh(logits / softcap) * softcap
        return logits[..., : text.unpadded_vocab_size][:, -1, :]

    def step(self, tokens):
        logits = self.logits(self.hidden(tokens))
        out = []
        for i in range(self.C):
            row = logits[i : i + 1]
            lp = row - mx.logsumexp(row, keepdims=True)
            out.append(mx.argmax(lp, axis=-1))
        return mx.concatenate(out)
