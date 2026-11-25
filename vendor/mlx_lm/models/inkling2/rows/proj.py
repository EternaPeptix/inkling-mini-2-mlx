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
import os
import weakref
import mlx.core as mx

from . import gemv as _gemv
from . import qmv as _qmv
from .glue import GLUE_HEADER

MAX_B = _qmv.MAX_B

# =============================================================================================== K4 / K9 (qmv)
# The tile is qmv.tile_source with a per-weight x pointer and an optional prologue that computes the x element on load:
# x = bf16(f32(a) * sigmoid(f32(g))), summed exactly like MLX's load_vector (8-bit: sum += x[i]).  With STG the whole
# threadgroup computes each 256-wide x block once into threadgroup memory (instead of every simdgroup recomputing it).
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
_STG_FILL = r"""
    if (STG) {
      threadgroup bfloat16_t* sb = sx + ((k / block_size) & 1) * RB * block_size;
      for (int idx = tid; idx < nb * block_size; idx += NSG * 32) {
        const int b = idx / block_size, j = idx % block_size;
        sb[idx] = static_cast<bfloat16_t>(glue_gate(static_cast<float>(xb[b * K + k + j]), static_cast<float>(gb[b * K + k + j])));
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""
_XLOAD_PRO = r"""
        if (STG) {
          const threadgroup bfloat16_t* p = sx + ((k / block_size) & 1) * RB * block_size + b * block_size + simd_lid * values_per_thread;
          U s = 0;
          for (int i = 0; i < values_per_thread; i++) { U v = p[i]; s += v; xtb[i] = v; }
          smb = s;
        } else {
          smb = proj_gate_load<values_per_thread>(x + b * K, gt + b * K, xtb);
        }
"""

_qh = {}


def _qheader(pro):
    if pro not in _qh:
        if pro:
            _qh[pro] = _qmv.base_header() + GLUE_HEADER + _PRO + _qmv.tile_source(_XLOAD_PRO, _STG_FILL)
        else:
            _qh[pro] = _qmv.base_header() + _qmv.tile_source(include_rout=True, include_ro=True)
    return _qh[pro]


_kernels = {}


def _qkernel(nw, xmap, pro, rout=False, ro=False):
    use_rout = rout and not pro
    use_ro = ro and not pro and not use_rout
    key = (nw, xmap, pro, use_rout, use_ro)
    k = _kernels.get(key)
    if k is not None:
        return k
    nx = max(xmap) + 1
    ins = [f"x{j}" for j in range(nx)] + (["gt"] if pro else [])
    for i in range(nw):
        ins += [f"w{i}", f"s{i}", f"b{i}"]
    lines = [
        "uint ty = threadgroup_position_in_grid.y;",
        "uint rb0 = threadgroup_position_in_grid.z * RB;",
        "uint sg = simdgroup_index_in_threadgroup;",
        "uint sl = thread_index_in_simdgroup;",
    ]
    fn = "rows_qmv_ro" if use_ro else ("rows_qmv_rout" if use_rout else "rows_qmv_tile")
    tail_t = "" if (use_rout or use_ro) else ", STG"
    if not pro:
        lines += [
            "const device uint32_t* wp; const device bfloat16_t* sp; const device bfloat16_t* bp;",
            "const device bfloat16_t* xp; device bfloat16_t* yp; int Ni; uint tyl;",
        ]
        prev = "0"
        for i in range(nw):
            end = f"{prev} + ((N{i} + RPS * NSG - 1) / (RPS * NSG))"
            if i == nw - 1:
                head = "if (true)" if i == 0 else "else"
            else:
                head = ("if" if i == 0 else "else if") + f" (ty < {end})"
            yoff = "+".join(f"N{j}" for j in range(i)) or "0"
            lines.append(
                f"{head} {{ wp = w{i}; sp = s{i}; bp = b{i}; xp = x{xmap[i]}; yp = y + B * ({yoff}); Ni = N{i}; tyl = ty - ({prev}); }}"
            )
            prev = end
        lines.append(
            f"{fn}<B, K, true, RPS, NSG, RB{tail_t}>(wp, sp, bp, xp, xp, yp, Ni, tyl, rb0, sg, sl, (threadgroup bfloat16_t*)0);"
        )
    else:
        if pro:
            lines.insert(0, "threadgroup bfloat16_t sx[2 * RB * 256];")
        prev = "0"
        for i in range(nw):
            end = f"{prev} + ((N{i} + RPS * NSG - 1) / (RPS * NSG))"
            if i == nw - 1:
                head = "if (true)" if i == 0 else "else"
            else:
                head = ("if" if i == 0 else "else if") + f" (ty < {end})"
            gt = "gt"
            yoff = "+".join(f"N{j}" for j in range(i)) or "0"
            lines.append(
                f"{head} {{ {fn}<B, K, (N{i} % (RPS * NSG)) != 0, RPS, NSG, RB{tail_t}>"
                f"(w{i}, s{i}, b{i}, x{xmap[i]}, {gt}, y + B * ({yoff}), N{i}, ty - ({prev}), rb0, sg, sl, sx); }}"
            )
            prev = end
    suf = "_pro" if pro else ("_ro" if use_ro else ("_rout" if use_rout else ""))
    k = mx.fast.metal_kernel(
        name="f3_qmv_x%s_n%d%s" % ("".join(map(str, xmap)), nw, suf),
        input_names=ins,
        output_names=["y"],
        source="\n".join(lines),
        header=_qheader(pro),
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
    tot_n = 0
    for w, _, _ in weights:
        tot_n += w.shape[0]
    pro = gate is not None
    rout = _qmv._use_rout(B, tot_n, len(weights), pro=pro)
    ro = _qmv._use_ro(B, tot_n, pro=pro, rout=rout)
    rps, nsg, rb, stg = _qmv._cfg(B, pro, N=tot_n, rout=rout, ro=ro)
    tmpl = [("B", B), ("K", K), ("RPS", rps), ("NSG", nsg), ("RB", rb)]
    if not ((rout or ro) and not pro):
        tmpl.append(("STG", stg))
    tile_w = rps * nsg
    kg = K // 64
    tiles = 0
    for i, (w, s, b) in enumerate(weights):
        N = w.shape[0]
        if w.shape[1] * 4 != K or s.shape != (N, kg) or b.shape != (N, kg) or s.dtype != mx.bfloat16:
            raise ValueError(f"weight {i}: bad shapes/dtype w{w.shape} s{s.shape}")
        args += [w, s, b]
        tmpl.append((f"N{i}", N))
        tiles += (N + tile_w - 1) // tile_w
    return _qkernel(len(weights), tuple(xmap), pro, rout=rout, ro=ro)(
        inputs=args,
        template=tmpl,
        grid=(32, nsg * tiles, (B + rb - 1) // rb),
        threadgroup=(32, nsg, 1),
        output_shapes=[(B * tot_n,)],
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
    return (m["weight"], m["scales"], m["biases"])


_qkv_cat_cache: dict = {}


def _aff8_ok(m):
    return (
        getattr(m, "mode", None) == "affine"
        and getattr(m, "bits", None) == 8
        and getattr(m, "group_size", None) == 64
        and "bias" not in m
        and m["scales"].dtype == mx.bfloat16
        and (m["weight"].shape[1] * 4) % 256 == 0
        and (m["weight"].shape[1] * 4) not in (64, 128)
    )


def _get_qkv_cat(a, single_x, full):
    """Build or reuse concatenated QKV weights for Attention `a`, invalidated immediately on weight mutation."""
    is_dict = isinstance(a, dict)
    if is_dict and not hasattr(a, "q_proj"):
        q_m, r_m, g_m, s_m, k_m, v_m = a["q_proj"], a["r_proj"], a["gate_proj"], a["sink_proj"], a["k_proj"], a["v_proj"]
        mod_pairs = [("q_proj", q_m), ("r_proj", r_m), ("gate", g_m), ("sinks", s_m), ("k_proj", k_m), ("v_proj", v_m)]
        if full:
            mod_pairs += [("k_wproj", a["k_sconv"]["weight_proj"]), ("v_wproj", a["v_sconv"]["weight_proj"])]
    else:
        mod_pairs = [("q_proj", a.q_proj), ("r_proj", a.r_proj), ("gate", a.gate_proj), ("sinks", a.sink_proj),
                     ("k_proj", a.k_proj), ("v_proj", a.v_proj)]
        if full:
            mod_pairs += [("k_wproj", a.k_sconv.weight_proj), ("v_wproj", a.v_sconv.weight_proj)]
    if not all(_aff8_ok(m) for _, m in mod_pairs):
        return None
    cur_refs = tuple(t for _, m in mod_pairs for t in (m["weight"], m["scales"], m["biases"]))
    key = (id(a), bool(single_x), bool(full))
    ent = _qkv_cat_cache.get(key)
    if ent is not None and len(ent[0]) == len(cur_refs) and all(x is y for x, y in zip(ent[0], cur_refs)):
        return ent[1]
    if len(_qkv_cat_cache) >= 512:
        _qkv_cat_cache.clear()
    if single_x and not full:
        grps = [mod_pairs]
    else:
        g0 = mod_pairs[:4]
        g1 = [mod_pairs[4], mod_pairs[6]] if full else [mod_pairs[4]]
        g2 = [mod_pairs[5], mod_pairs[7]] if full else [mod_pairs[5]]
        grps = [g0, g1, g2]
    cat_groups = []
    grp_widths = []
    fields = []
    for gi, grp in enumerate(grps):
        cw = grp[0][1]["weight"] if len(grp) == 1 else mx.contiguous(mx.concatenate([m["weight"] for _, m in grp], axis=0))
        cs = grp[0][1]["scales"] if len(grp) == 1 else mx.contiguous(mx.concatenate([m["scales"] for _, m in grp], axis=0))
        cb = grp[0][1]["biases"] if len(grp) == 1 else mx.contiguous(mx.concatenate([m["biases"] for _, m in grp], axis=0))
        cat_groups.append((cw, cs, cb))
        gw = cw.shape[0]
        grp_widths.append(gw)
        loc = 0
        for nm, m in grp:
            n = m["weight"].shape[0]
            fields.append((nm, gi, loc, gw, n))
            loc += n
    mx.eval(cat_groups)
    val = (tuple(cat_groups), tuple(grp_widths), tuple(fields))
    _qkv_cat_cache[key] = (cur_refs, val)
    if not is_dict or hasattr(a, "__weakref__"):
        try:
            weakref.finalize(a, _qkv_cat_cache.pop, key, None)
        except TypeError:
            pass
    return val


def qkv_rows(a, q_in, k_in, v_in, full=None, flat=False):
    """K4 for one Attention module `a`. q_in/k_in/v_in (B,D) bf16 (the same array may be passed for all three on sliding
    layers). Full-attention layers (a has k_sconv.weight_proj) also get k_wproj (from k_in) and v_wproj (from v_in).
    Returns dict: q_proj (B,3072), r_proj (B,384), gate (B,3072), sinks (B,24), k_proj (B,1024), v_proj (B,1024)
    [, k_wproj (B,32), v_wproj (B,32)].
    flat=True returns (buf, layout) instead: 1D buffer and [(name, offset, [stride,] N)]."""
    is_dict = isinstance(a, dict)
    if full is None:
        ks = a["k_sconv"] if is_dict else a.k_sconv
        full = ("weight_proj" in ks) if isinstance(ks, dict) else hasattr(ks, "weight_proj")
    if os.environ.get("LOOPLING_QKV_CAT", "1") not in ("0", ""):
        single_x = (not full) and (q_in is k_in) and (k_in is v_in)
        cat = _get_qkv_cat(a, single_x, full)
        if cat is not None:
            cat_groups, grp_widths, fields = cat
            B, K = q_in.shape
            if single_x:
                if B == 1:
                    w0, s0, b0 = cat_groups[0]
                    buf = mx.quantized_matmul(
                        q_in.reshape(1, 1, K), w0, scales=s0, biases=b0, transpose=True,
                        group_size=64, bits=8, mode="affine",
                    ).reshape(-1)
                else:
                    buf = _qmv.qmv_rows(q_in, *cat_groups[0]).reshape(-1)
                grp_bases = (0,)
            else:
                buf = _launch_q([q_in, k_in, v_in], list(cat_groups))
                grp_bases = (0, B * grp_widths[0], B * (grp_widths[0] + grp_widths[1]))
            if flat:
                lay = [(nm, grp_bases[gi] + loc, gw, nc) for nm, gi, loc, gw, nc in fields]
                return buf, lay
            out = {}
            for nm, gi, loc, gw, nc in fields:
                gb = grp_bases[gi]
                view = buf[gb : gb + B * gw].reshape(B, gw)[:, loc : loc + nc]
                out[nm] = view if (loc == 0 and gw == nc) else mx.contiguous(view)
            return out
    if is_dict:
        mods = [("q_proj", a["q_proj"], q_in), ("r_proj", a["r_proj"], q_in), ("gate", a["gate_proj"], q_in),
                ("sinks", a["sink_proj"], q_in), ("k_proj", a["k_proj"], k_in), ("v_proj", a["v_proj"], v_in)]
        if full:
            mods += [("k_wproj", a["k_sconv"]["weight_proj"], k_in), ("v_wproj", a["v_sconv"]["weight_proj"], v_in)]
    else:
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


def _gd_tm1_src(G, B):
    wsel = "\n".join(
        ("%s (g == %d) wr = w%d;" % ("if" if g == 0 else "else if", g, g)) if g < G - 1 else ("else wr = w%d;" % g)
        for g in range(G)
    )
    if G == 1:
        wsel = "const device bfloat16_t* wr = w0;"
    else:
        wsel = "const device bfloat16_t* wr;\n    " + wsel
    return f"""
    const int t = (int)threadgroup_position_in_grid.x;
    const int g = t / 64;
    const int row = t % 64;
    const int sgN = (int)simdgroup_index_in_threadgroup;
    const int thrN = (int)thread_index_in_simdgroup;
    const int bn0 = (sgN * 32 + thrN) * 4;
    {wsel}
    wr += (size_t)row * 4608 + bn0;
    const device bfloat16_t* xb0 = x + (size_t)g * 4608 + bn0;
    const size_t xbs = (size_t){G} * 4608;
    float result[{B}];
    for (int b = 0; b < {B}; b++) result[b] = 0.0f;
    for (int it = 0; it < 4; it++) {{
      const int off = it * 1024;
      const device bfloat16_t* wp = wr + off;
      bfloat4 wv = as_type<bfloat4>(((const device uint2*)wp)[0]);
      for (int b = 0; b < {B}; b++) {{
        const device bfloat16_t* xp = xb0 + (size_t)b * xbs + off;
        bfloat4 xv = as_type<bfloat4>(((const device uint2*)xp)[0]);
        result[b] += wv.x * static_cast<float>(xv.x);
        result[b] += wv.y * static_cast<float>(xv.y);
        result[b] += wv.z * static_cast<float>(xv.z);
        result[b] += wv.w * static_cast<float>(xv.w);
      }}
    }}
    if (bn0 + 4096 < 4608) {{
      const int off = 4096;
      const device bfloat16_t* wp = wr + off;
      bfloat4 wv = as_type<bfloat4>(((const device uint2*)wp)[0]);
      for (int b = 0; b < {B}; b++) {{
        const device bfloat16_t* xp = xb0 + (size_t)b * xbs + off;
        bfloat4 xv = as_type<bfloat4>(((const device uint2*)xp)[0]);
        result[b] += wv.x * static_cast<float>(xv.x);
        result[b] += wv.y * static_cast<float>(xv.y);
        result[b] += wv.z * static_cast<float>(xv.z);
        result[b] += wv.w * static_cast<float>(xv.w);
      }}
    }}
    for (int b = 0; b < {B}; b++) {{
      for (ushort sn = 16; sn >= 1; sn >>= 1) result[b] += simd_shuffle_down(result[b], sn);
    }}
    threadgroup float tgp[{B} * 8];
    if (thrN == 0) {{
      for (int b = 0; b < {B}; b++) tgp[b * 8 + sgN] = result[b];
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgN == 0 && thrN == 0) {{
      for (int b = 0; b < {B}; b++) {{
        float s = result[b];
        for (int k = 1; k < 8; k++) s += tgp[b * 8 + k];
        o[((size_t)b * {G} + g) * 64 + row] = static_cast<bfloat16_t>(s);
      }}
    }}
"""


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
    tm1 = (os.environ.get("LOOPLING_GD_TM1", "1") not in ("0", "")) and N == 64 and K == 4608 and B <= 16
    if tm1:
        key = ("tm1", B, G, N, K)
        kern = _k2.get(key)
        if kern is None:
            kern = _k2[key] = mx.fast.metal_kernel(
                name="f3_gate_down_tm1_b%d_g%d" % (B, G),
                input_names=["x"] + ["w%d" % g for g in range(G)],
                output_names=["o"],
                source=_gd_tm1_src(G, B),
                ensure_row_contiguous=True,
            )
        return kern(
            inputs=[x] + list(ws),
            grid=(G * 64 * 32, 8, 1),
            threadgroup=(32, 8, 1),
            output_shapes=[(B, G, N)],
            output_dtypes=[mx.bfloat16],
        )[0]
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
# forget/message_gate_proj (D=2304, 64) f32 at M=1: MLX gemv bm1 bn1 sm8 sn4 tm4 tn4 -> each quad of 4 lanes owns 1 row
# (K=64 split 4 lanes x 4 uint4/float4 loads), 8 rows per simdgroup, 4 simdgroups (32 output rows) per threadgroup;
# forget + message weights share one K-loop and reduce with shuffle_down(2), (1); lane (lane&3)==0 writes the epilogue.
_H3 = r"""
template <int B, int RB, int G, int D, int NSG>
METAL_FUNC void gate_mix_one(const device float* wf, const device float* wm, const device bfloat16_t* down, const int g,
                             const device float* s, const device float* msg, const int slot, device float* out,
                             const uint row0, const uint b0, const uint sg, const uint lane,
                             threadgroup bfloat16_t* dn_s) {
  const int nb = min(RB, B - (int)b0);
  const int row = int(row0 + (lane >> 2));
  const int qn = int(lane & 3) * 4;
  const uint tid = sg * 32 + lane;
  const uint tg_threads = uint(NSG * 32);
  for (uint idx = tid; idx < uint(nb * 16); idx += tg_threads) {
    uint b = idx >> 4;
    uint ch = (idx & 15u) * 4u;
    ((threadgroup uint2*)(dn_s + b * 64 + ch))[0] =
        ((const device uint2*)(down + ((b0 + b) * G + g) * 64 + ch))[0];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float fr[RB], mr[RB];
  for (int b = 0; b < RB; b++) {
    fr[b] = 0.0f;
    mr[b] = 0.0f;
  }
  const device float* wfr = wf + (size_t)row * 64 + qn;
  const device float* wmr = wm + (size_t)row * 64 + qn;
  for (int i = 0; i < 4; i++) {
    int off = i * 16;
    float4 ff = ((const device float4*)(wfr + off))[0];
    float4 mm = ((const device float4*)(wmr + off))[0];
    #pragma unroll
    for (int b = 0; b < RB; b++) {
      if (b < nb) {
        bfloat4 xv = as_type<bfloat4>(((const threadgroup uint2*)(dn_s + b * 64 + qn + off))[0]);
        float v0 = float(xv.x), v1 = float(xv.y), v2 = float(xv.z), v3 = float(xv.w);
        fr[b] += ff.x * v0; fr[b] += ff.y * v1; fr[b] += ff.z * v2; fr[b] += ff.w * v3;
        mr[b] += mm.x * v0; mr[b] += mm.y * v1; mr[b] += mm.z * v2; mr[b] += mm.w * v3;
      }
    }
  }
  for (int b = 0; b < RB; b++) {
    fr[b] += simd_shuffle_down(fr[b], 2);
    fr[b] += simd_shuffle_down(fr[b], 1);
    mr[b] += simd_shuffle_down(mr[b], 2);
    mr[b] += simd_shuffle_down(mr[b], 1);
  }
  if ((lane & 3) == 0) {
    for (int b = 0; b < RB; b++) {
      if (b < nb) {
        const int gb = (int)b0 + b;
        out[(gb * G + g) * D + row] = glue_msg_combine<0>(fr[b], mr[b], s[gb * D + row], msg[(gb * 4 + slot) * D + row]);
      }
    }
  }
}
"""

_k3 = {}
_GMIX_RB = int(os.environ.get("LOOPLING_GMIX_RB", "2"))


def gate_mix_rows(down, ss, msg, slots, wf, wm):
    """K3. down (B,G,64) bf16 (K2 output); ss = list of G arrays (B,D) f32 (the hidden state the gate mixes into; the same
    array may repeat); msg (B,4,D) f32; slots = G message indices (q,k,v = 0,1,2; mlp = 3); wf/wm = G forget/message
    weights (D,64) f32.  Returns (B,G,D) f32:  (1 - tanh(wf@f32(down))) * s + tanh(wm@f32(down)) * msg[slot], exactly the
    MessageGate epilogue (separate mul/add, no fma)."""
    B, G, R = down.shape
    D = wf[0].shape[0]
    if R != 64 or down.dtype != mx.bfloat16 or not 1 <= B <= MAX_B or len(ss) != G or len(wf) != G or len(wm) != G:
        raise ValueError("down must be (B,G,64) bf16; one s/wf/wm per gate")
    if D % 32 or (D != 2304 and _gemv._cfg(D, 64) != (1, 1, 8, 4, 4, 4)):
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
    nsg = 2 if (G == 1 and D % 16 == 0) else 4
    rows_tg = nsg * 8
    nt = D // rows_tg
    rb = min(B, max(1, _GMIX_RB))
    key = (B, rb, G, D, nsg, tuple(smap), tuple(slots))
    kern = _k3.get(key)
    if kern is None:
        lines = [
            f"threadgroup bfloat16_t dn_s[{rb} * 64];",
            "const uint t = threadgroup_position_in_grid.x;",
            f"const uint b0 = threadgroup_position_in_grid.y * {rb};",
            "const uint sg = simdgroup_index_in_threadgroup;",
            "const uint lane = thread_index_in_simdgroup;",
            f"const uint row0 = (t % {nt}) * {rows_tg} + sg * 8;",
        ]
        for g in range(G):
            if g < G - 1:
                head = ("if" if g == 0 else "else if") + f" (t < {(g + 1) * nt})"
            else:
                head = "else" if g else ""
            lines.append(
                "%s { gate_mix_one<%d, %d, %d, %d, %d>(wf%d, wm%d, down, %d, s%d, msg, %d, out, row0, b0, sg, lane, dn_s); }"
                % (head, B, rb, G, D, nsg, g, g, g, smap[g], slots[g])
            )
        kern = mx.fast.metal_kernel(
            name="f3_gate_mix_b%d_rb%d_g%d_n%d_s%s_m%s" % (B, rb, G, nsg, "".join(map(str, smap)), "".join(map(str, slots))),
            input_names=["down", "msg"] + ["s%d" % j for j in range(len(uniq))]
            + ["wf%d" % g for g in range(G)] + ["wm%d" % g for g in range(G)],
            output_names=["out"],
            header=GLUE_HEADER + _H3,
            source="\n".join(lines),
            ensure_row_contiguous=True,
        )
        _k3[key] = kern
    tg_t = nsg * 32
    return kern(
        inputs=[down, msg] + uniq + list(wf) + list(wm),
        grid=(G * nt * tg_t, (B + rb - 1) // rb, 1),
        threadgroup=(tg_t, 1, 1),
        output_shapes=[(B, G, D)],
        output_dtypes=[mx.float32],
    )[0]
