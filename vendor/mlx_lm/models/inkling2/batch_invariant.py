"""LOOPLING_BATCH_INVARIANT=1 (lane D): a T-token decode-time forward (2 <= T <= 8, B == 1, with cache) whose per-row
results are bitwise identical to T sequential 1-token forwards. Everything here is installed by install() as
class-level wrappers that fall through to the original code whenever the call is not such a forward.
LOOPLING_BI_OPS (comma list, default all) lets bench/bi_compare.py enable sources one by one."""
import math, os
import mlx.core as mx
import mlx.nn as nn

_MAXT = 8
_ORIG = {}
ACTIVE = False  # set by the caller only around a decode-time verify forward (prefill chunks of 2..8 must not take this path)


class active:
    def __enter__(self):
        global ACTIVE
        self.prev, ACTIVE = ACTIVE, True

    def __exit__(self, *a):
        global ACTIVE
        ACTIVE = self.prev


def enabled_flag():
    return os.environ.get("LOOPLING_BATCH_INVARIANT") == "1"


def _enabled(name):
    ops = os.environ.get("LOOPLING_BI_OPS")
    return ops is None or name in ops.split(",")


def _rows(x):
    """T if x is a (1, T, ...) decode-verify activation else 0."""
    return x.shape[1] if ACTIVE and x.ndim >= 3 and x.shape[0] == 1 and 2 <= x.shape[1] <= _MAXT else 0


def _qlinear(self, x):
    T = _rows(x)
    if not T or not _enabled("qmm"):
        return _ORIG["QL"](self, x)
    # (1,T,K) -> (T,1,K) against the weights broadcast (stride 0) over a batch of T: each row is its own M=1 problem
    # (the plain reshape is folded into M and takes the M>1 kernel)
    xr = x.reshape(T, 1, x.shape[-1])
    bc = lambda a: None if a is None else mx.broadcast_to(a[None], (T,) + a.shape)
    y = mx.quantized_matmul(xr, bc(self["weight"]), scales=bc(self["scales"]), biases=bc(self.get("biases")),
                            transpose=True, group_size=self.group_size, bits=self.bits, mode=self.mode)
    if "bias" in self:
        y = y + self["bias"]
    return y.reshape(1, T, -1)


def _linear(self, x):
    T = _rows(x)
    if not T or not _enabled("mm"):
        return _ORIG["L"](self, x)
    wt = self["weight"].T
    # M=1 gemv per row (a batched bf16 or float32 matmul takes the M>1 gemm kernel: bench/bi_compare.py, message-gate down_proj)
    y = mx.concatenate([x[:, j:j + 1] @ wt for j in range(T)], axis=1)
    if "bias" in self:
        y = y + self["bias"]
    return y.reshape(1, T, -1)


def _router_call(self, hidden_states):
    from mlx_lm.models.inkling2.layers import _ROUTER_SCORE_EPS, _use_kernels
    from mlx_lm.kernels.expert_routing import expert_routing
    T = _rows(hidden_states)
    if not T or not _use_kernels() or not _enabled("router"):
        return _ORIG["R"](self, hidden_states)
    h = hidden_states.astype(mx.float32)
    wt = self.weight.T
    logits = mx.concatenate([h[:, j:j + 1] @ wt for j in range(T)], axis=1)
    return expert_routing(logits, self.e_score_correction_bias, top_k=self.top_k, multiplier=1.0,
                          score_eps=_ROUTER_SCORE_EPS, route_scale=self.routed_scaling_factor)


def _conv_call(self, x):
    # depthwise conv over (1, K-1+T, C) -> T independent length-K windows, each producing one output like decode does
    K = self.weight.shape[1]
    L = x.shape[1]
    T = L - (K - 1)
    if not ACTIVE or x.shape[0] != 1 or not (2 <= T <= _MAXT) or not _enabled("conv"):
        return _ORIG["C"](self, x)
    w = mx.concatenate([x[:, j:j + K] for j in range(T)], axis=0)  # (T, K, C)
    y = _ORIG["C"](self, w)  # (T, 1, C)
    return y.reshape(1, T, -1)


def _qswitch(self, x, indices, sorted_indices=False):
    return _ORIG["QS"](self, x, indices, False)


def _experts_call(self, x, indices):
    # never take the sorted-expert path (it changes the gather_qmm kernel)
    if _rows(x) and _enabled("moe"):
        x = mx.expand_dims(x, (-2, -3))
        indices = mx.stop_gradient(indices)
        gate_up = self.gate_up_proj(x, indices, sorted_indices=False)
        x = self.act_fn(gate_up[..., 0::2], gate_up[..., 1::2])
        x = self.down_proj(x, indices, sorted_indices=False)
        return x.squeeze(-2)
    return _ORIG["E"](self, x, indices)


def _attend_bi(self, hidden_states, query_states, key_states, value_states, relative_states, cache, tau=None):
    """Attention._attend for decode-verify: elementwise prologue on all T rows, then the cache write and the 1-query
    attention are done row by row exactly as in 1-token decode (same cache layout, same SDPA vector kernel)."""
    from mlx_lm.models.inkling2.layers import unweighted_rms_norm
    from mlx_lm.models.inkling2 import fastdecode
    B, T, _ = hidden_states.shape
    gate = self.gate_proj(hidden_states)
    sinks = self.sink_proj(hidden_states)
    query_states = query_states.reshape(B, T, self.num_heads, -1)
    key_states = key_states.reshape(B, T, self.num_key_value_heads, -1)
    value_states = value_states.reshape(B, T, self.num_key_value_heads, -1)
    relative_states = relative_states.reshape(B, T, self.num_heads, -1)
    query_states = self.q_l2_clip(query_states)
    key_states = self.k_l2_clip(key_states)
    value_states = unweighted_rms_norm(value_states.astype(mx.float32), self.eps).astype(value_states.dtype)
    relative_states = self.rel_l2_clip(relative_states)
    if tau is not None:
        query_states = (query_states.astype(mx.float32) * tau).astype(query_states.dtype)
        relative_states = (relative_states.astype(mx.float32) * tau).astype(relative_states.dtype)
    position_bias = self.rel_logits_proj(relative_states)
    query_states = query_states.transpose(0, 2, 1, 3)
    key_states = key_states.transpose(0, 2, 1, 3)
    value_states = value_states.transpose(0, 2, 1, 3)
    outs = []
    rows = []
    shim = type("S", (), {"self_attn": self})()
    for j in range(T):
        kj, vj = key_states[:, :, j:j + 1], value_states[:, :, j:j + 1]
        rows.append((kj, vj))
        # the 1-token decode attention core (cache write + SDPA vector kernel, masks cached per cache state)
        outs.append(fastdecode._core(shim, query_states[:, :, j:j + 1], kj, vj, position_bias[:, j:j + 1], sinks[:, j:j + 1], cache))
    cache._bi_rows = rows  # for selfspec's rotating-cache rollback (replay of the kept rows)
    attn_output = mx.concatenate(outs, axis=1)  # each (B,1,H,D)
    attn_output = attn_output.reshape(B, T, -1).astype(mx.float32)
    attn_output = attn_output * mx.sigmoid(gate.astype(mx.float32))
    return self.o_proj(attn_output.astype(gate.dtype))


def _attend(self, hidden_states, query_states, key_states, value_states, relative_states, cache, tau=None):
    if ACTIVE and cache is not None and 2 <= hidden_states.shape[1] <= _MAXT and hidden_states.shape[0] == 1 and _enabled("attn"):
        return _attend_bi(self, hidden_states, query_states, key_states, value_states, relative_states, cache, tau)
    return _ORIG["A"](self, hidden_states, query_states, key_states, value_states, relative_states, cache, tau)


def install():
    if _ORIG:
        return
    from mlx_lm.layers import moe
    from mlx_lm.models.inkling2 import layers as L
    _ORIG.update(QL=nn.QuantizedLinear.__call__, L=nn.Linear.__call__, C=nn.Conv1d.__call__,
                 QS=moe.QuantizedSwitchLinear.__call__, E=L.Experts.__call__, R=L.TopKRouter.__call__, A=L.Attention._attend)
    nn.QuantizedLinear.__call__ = _qlinear
    nn.Linear.__call__ = _linear
    nn.Conv1d.__call__ = _conv_call
    L.Experts.__call__ = _experts_call
    L.TopKRouter.__call__ = _router_call
    L.Attention._attend = _attend
