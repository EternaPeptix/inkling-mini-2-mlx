"""Fused projection kernels (lane f3): K4 QKV, K9 OPROJ, K2 GATE-DOWN, K3 GATE-MIX of docs/FUSED.md.

Every output row is bitwise equal to the B=1 reference of fused_ref.visit_ref evaluated on that row alone; weights are read
once per launch.  Not wired into the decode path.  Notes and results: docs/FUSED_PROJ.md.

* qmv_rows_multi_x(xs, weights)            K4: several 8-bit affine weights, each with its own (B,K) input (inputs may alias)
* qkv_rows(attn_module, q_in, k_in, v_in)  K4 for one Attention module -> dict of q_proj, r_proj, gate, sinks, k_proj,
                                            v_proj [, k_wproj, v_wproj]
* oproj_rows(attn_out, gate, w, s, b)      K9: o_proj(bf16(f32(attn_out) * sigmoid(f32(gate))))
* gate_down_rows(x, ws)                    K2: x (B,G,4608) bf16, G weights (64,4608) bf16 -> (B,G,64) bf16
* gate_mix_rows(down, ss, msg, slots, wf, wm)  K3 -> (B,G,D) f32
"""
import mlx.core as mx

from . import gemv as _gemv
from . import qmv as _qmv
from .glue import GLUE_HEADER

MAX_B = _qmv.MAX_B

# =============================================================================================== K4 / K9 (qmv)
# The tile is qmv._TILE with (a) a per-weight x pointer (already a parameter) and (b) an optional prologue that computes
# the x element on load: x = bf16(f32(a) * sigmoid(f32(g))), summed exactly like MLX's load_vector (8-bit: sum += x[i]).
_TILE2 = _qmv._TILE
for _old, _new in [
    ("template <int B, int K, bool PR>\nMETAL_FUNC void rows_qmv_tile(", "template <int B, int K, bool PR, bool PRO>\nMETAL_FUNC void rows_qmv_tile2("),
    ("    const device bfloat16_t* x,\n    device bfloat16_t* y,", "    const device bfloat16_t* x,\n    const device bfloat16_t* gt,\n    device bfloat16_t* y,"),
    ("  x += simd_lid * values_per_thread;\n", "  x += simd_lid * values_per_thread;\n  gt += simd_lid * values_per_thread;\n"),
    ("    x += block_size;\n", "    x += block_size;\n    gt += block_size;\n"),
    ("      U sum = load_vector<bfloat16_t, U, values_per_thread, bits>(x + b * K, x_thread);",
     "      U sum = PRO ? proj_gate_load<values_per_thread>(x + b * K, gt + b * K, x_thread)\n"
     "                  : load_vector<bfloat16_t, U, values_per_thread, bits>(x + b * K, x_thread);"),
]:
    assert _TILE2.count(_old) == 1, _old
    _TILE2 = _TILE2.replace(_old, _new)

_PRO = r"""
// K9 prologue: x[i] = bf16(f32(a[i]) * sigmoid(f32(g[i]))) (layers.Attention._attend_post), then load_vector's 8-bit body.
template <int N>
inline float proj_gate_load(const device bfloat16_t* a, const device bfloat16_t* g, thread float* x_thread) {
  float sum = 0;
  for (int i = 0; i < N; i++) {
    bfloat16_t v = static_cast<bfloat16_t>(glue_gate(static_cast<float>(a[i]), static_cast<float>(g[i])));
    sum += v;
    x_thread[i] = v;
  }
  return sum;
}
"""

_qh = None


def _qheader():
    global _qh
    if _qh is None:
        base = _qmv._header()
        assert base.endswith(_qmv._TILE)
        _qh = base[: -len(_qmv._TILE)] + GLUE_HEADER + _PRO + _TILE2
    return _qh


_kernels = {}


def _qkernel(nw, xmap, pro):
    key = (nw, xmap, pro)
    k = _kernels.get(key)
    if k is not None:
        return k
    nx = max(xmap) + 1
    ins = [f"x{j}" for j in range(nx)] + (["gt"] if pro else [])
    for i in range(nw):
        ins += [f"w{i}", f"s{i}", f"b{i}"]
    lines = [
        "uint ty = threadgroup_position_in_grid.y;",
        "uint sg = simdgroup_index_in_threadgroup;",
        "uint sl = thread_index_in_simdgroup;",
    ]
    prev = "0"
    for i in range(nw):
        end = f"{prev} + ((N{i} + 7) / 8)"
        if i == nw - 1:
            head = "if (true)" if i == 0 else "else"
        else:
            head = ("if" if i == 0 else "else if") + f" (ty < {end})"
        gt = "gt" if pro else f"x{xmap[i]}"
        yoff = "+".join(f"N{j}" for j in range(i)) or "0"
        lines.append(
            f"{head} {{ rows_qmv_tile2<B, K, (N{i} % 8) != 0, {'true' if pro else 'false'}>"
            f"(w{i}, s{i}, b{i}, x{xmap[i]}, {gt}, y + B * ({yoff}), N{i}, ty - ({prev}), sg, sl); }}"
        )
        prev = end
    k = mx.fast.metal_kernel(
        name="f3_qmv_x%s_n%d%s" % ("".join(map(str, xmap)), nw, "_pro" if pro else ""),
        input_names=ins,
        output_names=["y"],
        source="\n".join(lines),
        header=_qheader(),
        ensure_row_contiguous=True,
    )
    _kernels[key] = k
    return k


def _launch_q(xs, weights, gate=None):
    if not xs or len(xs) != len(weights):
        raise ValueError("need one input per weight")
    B, K = xs[0].shape
    if not 1 <= B <= MAX_B:
        raise ValueError(f"B must be 1..{MAX_B}")
    if K % _qmv._K_ALIGN or K in (64, 128):
        raise ValueError("K must be a multiple of 256 and not 64/128 (MLX would pick another qmv kernel)")
    uniq, xmap = [], []
    for x in xs:
        if x.shape != (B, K) or x.dtype != mx.bfloat16:
            raise ValueError("every x must be (B, K) bfloat16 with the same B, K")
        for j, u in enumerate(uniq):
            if u is x:
                xmap.append(j)
                break
        else:
            xmap.append(len(uniq))
            uniq.append(x)
    args = list(uniq) + ([gate] if gate is not None else [])
    tmpl = [("B", B), ("K", K)]
    shapes, tiles = [], 0  # shapes: (B, N_i)
    for i, (w, s, b) in enumerate(weights):
        N = w.shape[0]
        if w.shape[1] * 4 != K or s.shape != (N, K // 64) or b.shape != (N, K // 64) or s.dtype != mx.bfloat16:
            raise ValueError(f"weight {i}: bad shapes/dtype w{w.shape} s{s.shape}")
        args += [w, s, b]
        tmpl.append((f"N{i}", N))
        shapes.append((B, N))
        tiles += (N + 7) // 8
    return _qkernel(len(weights), tuple(xmap), gate is not None)(
        inputs=args,
        template=tmpl,
        grid=(32, 2 * tiles, 1),
        threadgroup=(32, 2, 1),
        output_shapes=[(B * sum(n for _, n in shapes),)],
        output_dtypes=[mx.bfloat16],
    )[0]


def qmv_rows_multi_x(xs, weights, flat=False):
    """K4 core. xs: list of (B,K) bf16, one per weight (same array object may repeat); weights [(w, scales, biases)].
    One launch writing ONE flat output buffer
    (Metal allows 31 buffers; the full-attention QKV launch has 27 inputs), concatenated as B*N_0 | B*N_1 | ...; the returned
    (B, N_i) arrays are slices of it. Row b of output i == quantized_matmul(xs[i][b], weight i) at M=1."""
    fl = _launch_q(list(xs), list(weights))
    if flat:
        return fl
    B = xs[0].shape[0]
    outs, off = [], 0
    for w, _, _ in weights:
        n = w.shape[0]
        outs.append(fl[off:off + B * n].reshape(B, n))  # zero-copy views of the single output buffer
        off += B * n
    return outs


def _P(m):
    return (m.weight, m.scales, m.biases)


def qkv_rows(a, q_in, k_in, v_in, full=None, flat=False):
    """K4 for one Attention module `a`. q_in/k_in/v_in (B,D) bf16 (the same array may be passed for all three on sliding
    layers). Full-attention layers (a has k_sconv.weight_proj) also get k_wproj (from k_in) and v_wproj (from v_in).
    Returns dict: q_proj (B,3072), r_proj (B,384), gate (B,3072), sinks (B,24), k_proj (B,1024), v_proj (B,1024)
    [, k_wproj (B,32), v_wproj (B,32)].
    flat=True returns (buf, layout) instead: the single output buffer (B*sum(N),) and [(name, offset, N)] (what a fused
    consumer kernel should read; the dict form returns zero-copy slice views of the same buffer)."""
    if full is None:
        full = hasattr(a.k_sconv, "weight_proj")
    mods = [("q_proj", a.q_proj, q_in), ("r_proj", a.r_proj, q_in), ("gate", a.gate_proj, q_in),
            ("sinks", a.sink_proj, q_in), ("k_proj", a.k_proj, k_in), ("v_proj", a.v_proj, v_in)]
    if full:
        mods += [("k_wproj", a.k_sconv.weight_proj, k_in), ("v_wproj", a.v_sconv.weight_proj, v_in)]
    outs = qmv_rows_multi_x([m[2] for m in mods], [_P(m[1]) for m in mods], flat=flat)
    if flat:  # (flat (B*sum N,) bf16, [(name, element offset, N)]): output i is (B, N_i) row-major at its offset
        B, lay, off = q_in.shape[0], [], 0
        for m in mods:
            n = m[1].weight.shape[0]
            lay.append((m[0], off, n))
            off += B * n
        return outs, lay
    return {m[0]: o for m, o in zip(mods, outs)}


def oproj_rows(attn_out, gate, w, scales, biases):
    """K9. attn_out (B,3072) or (B,1,24,128) bf16, gate (B,3072) bf16 -> (B,2304) bf16;
    row b == o_proj(bf16(f32(attn_out[b]) * sigmoid(f32(gate[b])))) at M=1."""
    B = gate.shape[0]
    K = gate.shape[-1]
    if attn_out.size != B * K or attn_out.dtype != mx.bfloat16 or gate.dtype != mx.bfloat16:
        raise ValueError("attn_out and gate must be bf16 with B*K elements")
    a2 = attn_out.reshape(B, K)
    g2 = gate.reshape(B, K)
    return _launch_q([a2], [(w, scales, biases)], gate=g2).reshape(B, w.shape[0])


# =============================================================================================== K2 (gemv rows)
_H2 = _gemv._HEADER
for _old, _new in [
    ("METAL_FUNC void gemv_rows_run(", "METAL_FUNC void gemv_rows_run2("),
    ("    const int K,\n    const int N,\n", "    const int K,\n    const int N,\n    const int XS,\n    const int OS,\n"),
    ("in_vec[b * K + bn + tn]", "in_vec[b * XS + bn + tn]"),
    ("out_vec[b * N + out_row + tm]", "out_vec[b * OS + out_row + tm]"),
]:
    assert _H2.count(_old) >= 1, _old
    _H2 = _H2.replace(_old, _new)

_k2 = {}


def gate_down_rows(x, ws):
    """K2. x (B,G,K) bf16 (row b, gate g = the 4608-vector of gate g), ws = G weights (64,K) bf16 as stored
    -> (B,G,64) bf16, == (x[b,g] @ ws[g].T) at M=1 for every (b,g); one launch."""
    B, G, K = x.shape
    if x.dtype != mx.bfloat16 or len(ws) != G or not 1 <= B <= MAX_B:
        raise ValueError("x must be (B,G,K) bf16 with one weight per gate, B 1..16")
    N = ws[0].shape[0]
    for w in ws:
        if w.shape != (N, K) or w.dtype != mx.bfloat16:
            raise ValueError("weights must all be (N,K) bf16")
    cfg = _gemv._cfg(N, K)
    bm, bn, sm, sn, tm, tn = cfg
    blockM = bm * sm * tm
    nt = (N + blockM - 1) // blockM
    key = (B, G, N, K, cfg)
    kern = _k2.get(key)
    if kern is None:
        tg_floats = B * bn * (blockM + tm) if bn > 1 else 1
        lines = ["threadgroup float tgp_mem[%d];" % tg_floats, "const uint t = threadgroup_position_in_grid.x;"]
        for g in range(G):
            lines.append(
                "if (t < %d) { gemv_rows_run2<bfloat16_t, %d, %d, %d, %d, %d, %d, %d>(w%d, x + %d, o + %d, %d, %d, %d, %d, tgp_mem, t - %d, "
                "simdgroup_index_in_threadgroup, thread_index_in_simdgroup); return; }"
                % ((g + 1) * nt, B, bm, bn, sm, sn, tm, tn, g, g * K, g * N, K, N, G * K, G * N, g * nt)
            )
        kern = mx.fast.metal_kernel(
            name="f3_gate_down_b%d_g%d_n%d_k%d" % (B, G, N, K),
            input_names=["x"] + ["w%d" % g for g in range(G)],
            output_names=["o"],
            header=_H2,
            source="\n".join(lines),
            ensure_row_contiguous=True,
        )
        _k2[key] = kern
    return kern(
        inputs=[x] + list(ws),
        grid=(G * nt * 32, bn, bm),
        threadgroup=(32, bn, bm),
        output_shapes=[(B, G, N)],
        output_dtypes=[mx.bfloat16],
    )[0]


# =============================================================================================== K3 (gate mix)
# forget/message_gate_proj (D=2304, 64) f32 at M=1: MLX gemv bm1 bn1 sm8 sn4 tm4 tn4 -> a 32-thread threadgroup owns 32
# output rows; the 4 lanes of a quad share 4 rows and reduce with shuffle_down(2), (1); lane quad%4==0 holds the result.
_H3 = r"""
// GEMVKernel::run for the (2304,64) f32 weight; x is the bf16 gate_down row cast to f32 on load (== states.astype(f32)).
// out: res[b*4+tm] (valid in lanes with lane%4==0).  Same expression order as rows/gemv.py gemv_rows_run.
template <int B>
METAL_FUNC void gate_gemv64(const device float* mat, const device bfloat16_t* xin, const int XS, const uint row0, const uint lane,
                            thread float* res) {
  constexpr int K = 64, TM = 4, TN = 4, SN = 4;
  constexpr int blockN = SN * TN;
  for (int i = 0; i < B * TM; i++) res[i] = 0.0f;
  float v_coeff[B][TN];
  float inter[TN];
  int bn = int(lane % SN) * TN;
  mat += size_t(row0 + (lane / SN) * TM) * K;
  for (int i = 0; i < K / blockN; ++i) {
    for (int b = 0; b < B; b++)
      for (int tn = 0; tn < TN; tn++) v_coeff[b][tn] = static_cast<float>(xin[b * XS + bn + tn]);
    int mat_offset = 0;
    for (int tm = 0; tm < TM; tm++) {
      for (int tn = 0; tn < TN; tn++) inter[tn] = mat[mat_offset + bn + tn];
      for (int b = 0; b < B; b++)
        for (int tn = 0; tn < TN; tn++) res[b * TM + tm] += inter[tn] * v_coeff[b][tn];
      mat_offset += K;
    }
    bn += blockN;
  }
  for (int b = 0; b < B; b++)
    for (int tm = 0; tm < TM; tm++)
      for (ushort sn = (SN / 2); sn >= 1; sn >>= 1) res[b * TM + tm] += simd_shuffle_down(res[b * TM + tm], sn);
}

template <int B, int G, int D>
METAL_FUNC void gate_mix_one(const device float* wf, const device float* wm, const device bfloat16_t* down, const int g,
                             const device float* s, const device float* msg, const int slot, device float* out,
                             const uint row0, const uint lane) {
  float fr[B * 4], mr[B * 4];
  gate_gemv64<B>(wf, down + g * 64, G * 64, row0, lane, fr);
  gate_gemv64<B>(wm, down + g * 64, G * 64, row0, lane, mr);
  if (lane % 4 == 0) {
    for (int tm = 0; tm < 4; tm++) {
      const int d = int(row0 + (lane / 4) * 4) + tm;
      for (int b = 0; b < B; b++) {
        out[(b * G + g) * D + d] = glue_msg_combine<0>(fr[b * 4 + tm], mr[b * 4 + tm], s[b * D + d], msg[(b * 4 + slot) * D + d]);
      }
    }
  }
}
"""

_k3 = {}


def gate_mix_rows(down, ss, msg, slots, wf, wm):
    """K3. down (B,G,64) bf16 (K2 output); ss = list of G arrays (B,D) f32 (the hidden state the gate mixes into; the same
    array may repeat); msg (B,4,D) f32; slots = G message indices (q,k,v = 0,1,2; mlp = 3); wf/wm = G forget/message
    weights (D,64) f32.  Returns (B,G,D) f32:  (1 - tanh(wf@f32(down))) * s + tanh(wm@f32(down)) * msg[slot], exactly the
    MessageGate epilogue (separate mul/add, no fma)."""
    B, G, R = down.shape
    D = wf[0].shape[0]
    if R != 64 or down.dtype != mx.bfloat16 or not 1 <= B <= MAX_B or len(ss) != G or len(wf) != G or len(wm) != G:
        raise ValueError("down must be (B,G,64) bf16; one s/wf/wm per gate")
    if _gemv._cfg(D, 64) != (1, 1, 8, 4, 4, 4) or D % 32:
        raise ValueError("gate_mix_rows is specialised for D % 32 == 0 and MLX's bm1 bn1 sm8 sn4 tm4 tn4 gemv config")
    for w in list(wf) + list(wm):
        if w.shape != (D, 64) or w.dtype != mx.float32:
            raise ValueError("forget/message weights must be (D,64) f32")
    if msg.shape != (B, 4, D) or msg.dtype != mx.float32 or any(s.shape != (B, D) or s.dtype != mx.float32 for s in ss):
        raise ValueError("msg must be (B,4,D) f32 and each s (B,D) f32")
    uniq, smap = [], []
    for s in ss:
        for j, u in enumerate(uniq):
            if u is s:
                smap.append(j)
                break
        else:
            smap.append(len(uniq))
            uniq.append(s)
    nt = D // 32
    key = (B, G, D, tuple(smap), tuple(slots))
    kern = _k3.get(key)
    if kern is None:
        lines = ["const uint t = threadgroup_position_in_grid.x;", "const uint lane = thread_index_in_simdgroup;",
                 f"const uint row0 = (t % {nt}) * 32;"]
        for g in range(G):
            if g < G - 1:
                head = ("if" if g == 0 else "else if") + f" (t < {(g + 1) * nt})"
            else:
                head = "else" if g else ""
            lines.append(
                "%s { gate_mix_one<%d, %d, %d>(wf%d, wm%d, down, %d, s%d, msg, %d, out, row0, lane); }"
                % (head, B, G, D, g, g, g, smap[g], slots[g])
            )
        kern = mx.fast.metal_kernel(
            name="f3_gate_mix_b%d_g%d_s%s_m%s" % (B, G, "".join(map(str, smap)), "".join(map(str, slots))),
            input_names=["down", "msg"] + ["s%d" % j for j in range(len(uniq))]
            + ["wf%d" % g for g in range(G)] + ["wm%d" % g for g in range(G)],
            output_names=["out"],
            header=GLUE_HEADER + _H3,
            source="\n".join(lines),
            ensure_row_contiguous=True,
        )
        _k3[key] = kern
    return kern(
        inputs=[down, msg] + uniq + list(wf) + list(wm),
        grid=(G * nt * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(B, G, D)],
        output_dtypes=[mx.float32],
    )[0]
