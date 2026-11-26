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
import math
import os
import weakref
import mlx.core as mx
import numpy as np

from . import gemv as _gemv
from . import mid as _mid
from . import qmv as _qmv
from .glue import GLUE_HEADER, flit

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


def _qkernel(nw, xmap, pro, rout=False, ro=False, stg_x=False, f32x=False):
    use_rout = rout and not pro
    use_ro = ro and not pro and not use_rout
    use_stg_x = bool(stg_x and use_rout and not use_ro)
    use_f32x = bool(f32x and use_rout and not use_ro and not use_stg_x)
    key = (nw, xmap, pro, use_rout, use_ro, use_stg_x, use_f32x)
    k = _kernels.get(key)
    if k is not None:
        return k
    nx = max(xmap) + 1
    ins = [f"x{j}" for j in range(nx)] + (["gt"] if pro else [])
    for i in range(nw):
        ins += [f"w{i}", f"s{i}", f"b{i}"]
    lines = [
        "uint ty = threadgroup_position_in_grid.y;",
        "uint rb0 = threadgroup_position_in_grid.x * RB;",
        "uint sg = simdgroup_index_in_threadgroup;",
        "uint sl = thread_index_in_simdgroup;",
    ]
    fn = "rows_qmv_ro" if use_ro else ("rows_qmv_rout" if use_rout else "rows_qmv_tile")
    tail_t = (", true" if use_stg_x else (", false, false" if use_f32x else "")) if use_rout else ("" if use_ro else ", STG")
    if not pro:
        if use_stg_x:
            lines.insert(0, "threadgroup uint32_t sx_raw[RB * 160]; threadgroup bfloat16_t* sx = (threadgroup bfloat16_t*)sx_raw;")
        sx_arg = "sx" if use_stg_x else "(threadgroup bfloat16_t*)0"
        prev = "0"
        for i in range(nw):
            end = f"{prev} + ((N{i} + RPS * NSG - 1) / (RPS * NSG))"
            if i == nw - 1:
                head = "if (true)" if i == 0 else "else"
            else:
                head = ("if" if i == 0 else "else if") + f" (ty < {end})"
            yoff = "+".join(f"N{j}" for j in range(i)) or "0"
            lines.append(
                f"{head} {{ {fn}<B, K, (N{i} % (RPS * NSG)) != 0, RPS, NSG, RB{tail_t}>"
                f"(w{i}, s{i}, b{i}, x{xmap[i]}, x{xmap[i]}, y + B * ({yoff}), N{i}, ty - ({prev}), rb0, sg, sl, {sx_arg}); }}"
            )
            prev = end
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
    suf = "_pro" if pro else ("_ro" if use_ro else ("_rout_sx" if use_stg_x else ("_rout_fx" if use_f32x else ("_rout" if use_rout else ""))))
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
    f32x = _qmv._use_rout_f32x(B, tot_n, len(weights), pro=pro) if rout else False
    ro = _qmv._use_ro(B, tot_n, pro=pro, rout=rout)
    stg_x = _qmv._use_stg_x(B, tot_n, rout, pro=pro)
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
    return _qkernel(len(weights), tuple(xmap), pro, rout=rout, ro=ro, stg_x=stg_x, f32x=f32x)(
        inputs=args,
        template=tmpl,
        grid=(((B + rb - 1) // rb) * 32, nsg * tiles, 1),
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
_QKV_CAT = os.environ.get("LOOPLING_QKV_CAT", "1") not in ("0", "")
_GD_TM1 = os.environ.get("LOOPLING_GD_TM1", "1") not in ("0", "")
_OUT_DT_BF = [mx.bfloat16]
_OUT_DT_F32_2 = [mx.float32, mx.float32]


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
    key = (id(a), bool(single_x), bool(full))
    ent = _qkv_cat_cache.get(key)
    is_dict = isinstance(a, dict)
    if is_dict and not hasattr(a, "q_proj"):
        q_m, r_m, g_m, s_m, k_m, v_m = a["q_proj"], a["r_proj"], a["gate_proj"], a["sink_proj"], a["k_proj"], a["v_proj"]
        if full:
            kw_m, vw_m = a["k_sconv"]["weight_proj"], a["v_sconv"]["weight_proj"]
            cur_refs = (
                q_m["weight"], q_m["scales"], q_m["biases"],
                r_m["weight"], r_m["scales"], r_m["biases"],
                g_m["weight"], g_m["scales"], g_m["biases"],
                s_m["weight"], s_m["scales"], s_m["biases"],
                k_m["weight"], k_m["scales"], k_m["biases"],
                v_m["weight"], v_m["scales"], v_m["biases"],
                kw_m["weight"], kw_m["scales"], kw_m["biases"],
                vw_m["weight"], vw_m["scales"], vw_m["biases"],
            )
            if ent is not None and ent[0] is not None:
                er = ent[0]
                if (len(er) == 24
                    and er[0] is cur_refs[0] and er[1] is cur_refs[1] and er[2] is cur_refs[2]
                    and er[3] is cur_refs[3] and er[4] is cur_refs[4] and er[5] is cur_refs[5]
                    and er[6] is cur_refs[6] and er[7] is cur_refs[7] and er[8] is cur_refs[8]
                    and er[9] is cur_refs[9] and er[10] is cur_refs[10] and er[11] is cur_refs[11]
                    and er[12] is cur_refs[12] and er[13] is cur_refs[13] and er[14] is cur_refs[14]
                    and er[15] is cur_refs[15] and er[16] is cur_refs[16] and er[17] is cur_refs[17]
                    and er[18] is cur_refs[18] and er[19] is cur_refs[19] and er[20] is cur_refs[20]
                    and er[21] is cur_refs[21] and er[22] is cur_refs[22] and er[23] is cur_refs[23]):
                    return ent[1]
            mod_pairs = [("q_proj", q_m), ("r_proj", r_m), ("gate", g_m), ("sinks", s_m),
                         ("k_proj", k_m), ("v_proj", v_m), ("k_wproj", kw_m), ("v_wproj", vw_m)]
        else:
            cur_refs = (
                q_m["weight"], q_m["scales"], q_m["biases"],
                r_m["weight"], r_m["scales"], r_m["biases"],
                g_m["weight"], g_m["scales"], g_m["biases"],
                s_m["weight"], s_m["scales"], s_m["biases"],
                k_m["weight"], k_m["scales"], k_m["biases"],
                v_m["weight"], v_m["scales"], v_m["biases"],
            )
            if ent is not None and ent[0] is not None:
                er = ent[0]
                if (len(er) == 18
                    and er[0] is cur_refs[0] and er[1] is cur_refs[1] and er[2] is cur_refs[2]
                    and er[3] is cur_refs[3] and er[4] is cur_refs[4] and er[5] is cur_refs[5]
                    and er[6] is cur_refs[6] and er[7] is cur_refs[7] and er[8] is cur_refs[8]
                    and er[9] is cur_refs[9] and er[10] is cur_refs[10] and er[11] is cur_refs[11]
                    and er[12] is cur_refs[12] and er[13] is cur_refs[13] and er[14] is cur_refs[14]
                    and er[15] is cur_refs[15] and er[16] is cur_refs[16] and er[17] is cur_refs[17]):
                    return ent[1]
            mod_pairs = [("q_proj", q_m), ("r_proj", r_m), ("gate", g_m), ("sinks", s_m), ("k_proj", k_m), ("v_proj", v_m)]
    else:
        mod_pairs = [("q_proj", a.q_proj), ("r_proj", a.r_proj), ("gate", a.gate_proj), ("sinks", a.sink_proj),
                     ("k_proj", a.k_proj), ("v_proj", a.v_proj)]
        if full:
            mod_pairs += [("k_wproj", a.k_sconv.weight_proj), ("v_wproj", a.v_sconv.weight_proj)]
        cur_refs = tuple(t for _, m in mod_pairs for t in (m["weight"], m["scales"], m["biases"]))
        if ent is not None and len(ent[0]) == len(cur_refs) and all(x is y for x, y in zip(ent[0], cur_refs)):
            return ent[1]
    if not all(_aff8_ok(m) for _, m in mod_pairs):
        return None
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
    flat_w = [t for g in cat_groups for t in g]
    val = (tuple(cat_groups), tuple(grp_widths), tuple(fields), flat_w, {})
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
    if _QKV_CAT:
        single_x = (not full) and (q_in is k_in) and (k_in is v_in)
        cat = _get_qkv_cat(a, single_x, full)
        if cat is not None:
            cat_groups, grp_widths, fields, flat_w, b_cache = cat
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
            elif 1 <= B <= MAX_B and K % _qmv._K_ALIGN == 0 and K not in (64, 128) and q_in.dtype == mx.bfloat16 and k_in.dtype == mx.bfloat16 and v_in.dtype == mx.bfloat16:
                xmap = (0, 0, 0) if (q_in is k_in and k_in is v_in) else ((0, 1, 1) if k_in is v_in else ((0, 0, 1) if q_in is k_in else ((0, 1, 0) if q_in is v_in else (0, 1, 2))))
                lkey = ("lq", B, K, xmap)
                lmeta = b_cache.get(lkey)
                if lmeta is None:
                    tot_n = sum(grp_widths)
                    nw = len(cat_groups)
                    rout = _qmv._use_rout(B, tot_n, nw, pro=False)
                    f32x = _qmv._use_rout_f32x(B, tot_n, nw, pro=False) if rout else False
                    ro = _qmv._use_ro(B, tot_n, pro=False, rout=rout)
                    stg_x = _qmv._use_stg_x(B, tot_n, rout, pro=False)
                    rps, nsg, rb, stg = _qmv._cfg(B, False, N=tot_n, rout=rout, ro=ro)
                    tmpl = [("B", B), ("K", K), ("RPS", rps), ("NSG", nsg), ("RB", rb)]
                    if not (rout or ro):
                        tmpl.append(("STG", stg))
                    tw = rps * nsg
                    tiles = 0
                    for i, gw in enumerate(grp_widths):
                        tmpl.append((f"N{i}", gw))
                        tiles += (gw + tw - 1) // tw
                    kern = _qkernel(nw, xmap, False, rout=rout, ro=ro, stg_x=stg_x, f32x=f32x)
                    grid = (((B + rb - 1) // rb) * 32, nsg * tiles, 1)
                    tg = (32, nsg, 1)
                    out_shp = [(B * tot_n,)]
                    grp_bases = (0, B * grp_widths[0], B * (grp_widths[0] + grp_widths[1]))
                    lay = [(nm, grp_bases[gi] + loc, gw, nc) for nm, gi, loc, gw, nc in fields]
                    lmeta = b_cache[lkey] = (kern, tmpl, grid, tg, out_shp, lay, grp_bases)
                kern, tmpl, grid, tg, out_shp, lay, grp_bases = lmeta
                if xmap == (0, 1, 2):
                    x_in = [q_in, k_in, v_in]
                elif xmap == (0, 0, 0):
                    x_in = [q_in]
                elif xmap == (0, 1, 1):
                    x_in = [q_in, k_in]
                elif xmap == (0, 0, 1):
                    x_in = [q_in, v_in]
                else:
                    x_in = [q_in, k_in]
                buf = kern(
                    inputs=x_in + flat_w,
                    template=tmpl,
                    grid=grid,
                    threadgroup=tg,
                    output_shapes=out_shp,
                    output_dtypes=_OUT_DT_BF,
                )[0]
                if flat:
                    return buf, lay
                out = {}
                for nm, gi, loc, gw, nc in fields:
                    gb = grp_bases[gi]
                    view = buf[gb : gb + B * gw].reshape(B, gw)[:, loc : loc + nc]
                    out[nm] = view if (loc == 0 and gw == nc) else mx.contiguous(view)
                return out
            else:
                buf = _launch_q([q_in, k_in, v_in], list(cat_groups))
            if flat:
                lay = b_cache.get(("lay", B))
                if lay is None:
                    grp_bases = (0,) if single_x else (0, B * grp_widths[0], B * (grp_widths[0] + grp_widths[1]))
                    lay = b_cache[("lay", B)] = [(nm, grp_bases[gi] + loc, gw, nc) for nm, gi, loc, gw, nc in fields]
                return buf, lay
            grp_bases = (0,) if single_x else (0, B * grp_widths[0], B * (grp_widths[0] + grp_widths[1]))
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


# =============================================================================================== K3b+K4 fused (qkv from mixed)
_QNORM_SRC = r"""
inline float proj_inv_from_part_q(const device float* p, float eps, uint lane) {
  GL_NOFMA
  float pv = (lane < 18u) ? (p[lane * 2u] + p[lane * 2u + 1u]) : 0.0f;
  float tot = simd_sum(pv);
  return metal::precise::rsqrt(tot / 2304.0f + eps);
}

inline float proj_norm_load8(const device float* mp, float inv, bfloat4 lw0, bfloat4 lw1, thread float* xt) {
  float4 m0 = ((const device float4*)mp)[0];
  float4 m1 = ((const device float4*)(mp + 4))[0];
  bfloat16_t b0 = bfloat16_t(glue_rms_apply<float>(m0.x, inv, lw0.x));
  bfloat16_t b1 = bfloat16_t(glue_rms_apply<float>(m0.y, inv, lw0.y));
  bfloat16_t b2 = bfloat16_t(glue_rms_apply<float>(m0.z, inv, lw0.z));
  bfloat16_t b3 = bfloat16_t(glue_rms_apply<float>(m0.w, inv, lw0.w));
  bfloat16_t b4 = bfloat16_t(glue_rms_apply<float>(m1.x, inv, lw1.x));
  bfloat16_t b5 = bfloat16_t(glue_rms_apply<float>(m1.y, inv, lw1.y));
  bfloat16_t b6 = bfloat16_t(glue_rms_apply<float>(m1.z, inv, lw1.z));
  bfloat16_t b7 = bfloat16_t(glue_rms_apply<float>(m1.w, inv, lw1.w));
  xt[0] = float(b0); xt[1] = float(b1); xt[2] = float(b2); xt[3] = float(b3);
  xt[4] = float(b4); xt[5] = float(b5); xt[6] = float(b6); xt[7] = float(b7);
  float s = 0.0f;
  s += xt[0]; s += xt[1]; s += xt[2]; s += xt[3];
  s += xt[4]; s += xt[5]; s += xt[6]; s += xt[7];
  return s;
}

template <int B, int K, bool PR, int RPS, int NSG, int RB, bool USE_TILE>
METAL_FUNC void rows_qmv_norm_run(
    const device uint32_t* w,
    const device bfloat16_t* scales,
    const device bfloat16_t* biases,
    const device float* mixed,
    const device float* part,
    const device bfloat16_t* ln_w,
    const int gi,
    const float eps_ln,
    device bfloat16_t* y,
    const int N,
    const uint ty,
    const uint b0,
    const uint simd_gid,
    const uint simd_lid) {
  constexpr int group_size = 64;
  constexpr int bits = 8;
  constexpr int packs_per_thread = 2;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;
  static_assert(K == 2304 && values_per_thread == 8, "rows_qmv_norm_run expects K=2304, VPT=8");
  constexpr bool FULL = (B % RB) == 0;

  const int out_row = int(ty) * (NSG * RPS) + int(simd_gid) * RPS;
  int last_row = RPS - 1;
  if (PR) {
    if (out_row >= N) return;
    last_row = min(last_row, N - 1 - out_row);
  }

  typedef float U;
  const int nb = FULL ? RB : min(RB, B - int(b0));
  thread float inv[RB];
  const device float* pp0 = part + (size_t(b0) * 3u + uint(gi)) * 36u;
  _Pragma("clang loop unroll(full)")
  for (int b = 0; b < RB; b++) {
    if (FULL || b < nb) {
      inv[b] = proj_inv_from_part_q(pp0 + size_t(b) * 108u, eps_ln, simd_lid);
    } else {
      inv[b] = 0.0f;
    }
  }

  thread U result[RB][RPS];
  for (int b = 0; b < RB; b++)
    for (int r = 0; r < RPS; r++) result[b][r] = 0;

  const int in_vec_size_w = K * bytes_per_pack / pack_factor;
  const int in_vec_size_g = K / group_size;
  const device uint8_t* ws = (const device uint8_t*)w + size_t(out_row) * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales += size_t(out_row) * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += size_t(out_row) * in_vec_size_g + simd_lid / scale_step_per_thread;
  const device float* mp = mixed + (size_t(b0) * 3u + uint(gi)) * size_t(K) + simd_lid * values_per_thread;
  const device bfloat16_t* lw = ln_w + simd_lid * values_per_thread;
  y += size_t(b0) * N + out_row;

  for (int k = 0; k < K; k += block_size) {
    uint4 lwpk = ((const device uint4*)lw)[0];
    bfloat4 l0 = as_type<bfloat4>(lwpk.xy);
    bfloat4 l1 = as_type<bfloat4>(lwpk.zw);
    if constexpr (USE_TILE) {
      thread uint2 wr[RPS];
      thread U sc[RPS];
      thread U bi[RPS];
      _Pragma("clang loop unroll(full)")
      for (int row = 0; row < RPS; row++) {
        int src = PR ? min(row, last_row) : row;
        wr[row] = ((const device uint2*)(ws + size_t(src) * in_vec_size_w))[0];
        sc[row] = scales[size_t(src) * in_vec_size_g];
        bi[row] = biases[size_t(src) * in_vec_size_g];
      }
      _Pragma("clang loop unroll(full)")
      for (int b = 0; b < RB; b++) {
        if (FULL || b < nb) {
          thread U xtb[8];
          U smb = proj_norm_load8(mp + size_t(b) * (3u * size_t(K)), inv[b], l0, l1, xtb);
          _Pragma("clang loop unroll(full)")
          for (int row = 0; row < RPS; row++) {
            float4 w0 = float4(as_type<uchar4>(wr[row].x));
            float4 w1 = float4(as_type<uchar4>(wr[row].y));
            U accum = 0;
            accum += xtb[0] * w0.x; accum += xtb[1] * w0.y; accum += xtb[2] * w0.z; accum += xtb[3] * w0.w;
            accum += xtb[4] * w1.x; accum += xtb[5] * w1.y; accum += xtb[6] * w1.z; accum += xtb[7] * w1.w;
            result[b][row] += sc[row] * accum + smb * bi[row];
          }
        }
      }
    } else {
      thread U xtb[RB][8];
      thread U smb[RB];
      _Pragma("clang loop unroll(full)")
      for (int b = 0; b < RB; b++) {
        if (FULL || b < nb) {
          smb[b] = proj_norm_load8(mp + b * (3 * K), inv[b], l0, l1, xtb[b]);
        }
      }
      _Pragma("clang loop unroll(full)")
      for (int row = 0; row < RPS; row++) {
        int src = PR ? min(row, last_row) : row;
        uint2 wr = ((const device uint2*)(ws + src * in_vec_size_w))[0];
        U sc = scales[src * in_vec_size_g];
        U bi = biases[src * in_vec_size_g];
        float4 w0 = float4(as_type<uchar4>(wr.x));
        float4 w1 = float4(as_type<uchar4>(wr.y));
        _Pragma("clang loop unroll(full)")
        for (int b = 0; b < RB; b++) {
          if (FULL || b < nb) {
            U accum = 0;
            accum += xtb[b][0] * w0.x; accum += xtb[b][1] * w0.y; accum += xtb[b][2] * w0.z; accum += xtb[b][3] * w0.w;
            accum += xtb[b][4] * w1.x; accum += xtb[b][5] * w1.y; accum += xtb[b][6] * w1.z; accum += xtb[b][7] * w1.w;
            result[b][row] += sc * accum + smb[b] * bi;
          }
        }
      }
    }
    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    biases += block_size / group_size;
    mp += block_size;
    lw += block_size;
  }

  _Pragma("clang loop unroll(full)")
  for (int b = 0; b < RB; b++) {
    _Pragma("clang loop unroll(full)")
    for (int row = 0; row < RPS; row++) {
      U r = simd_sum(result[b][row]);
      if (simd_lid == 0 && row <= last_row && (FULL || b < nb)) {
        y[size_t(b) * N + row] = static_cast<bfloat16_t>(r);
      }
    }
  }
}

template <int B, int K, bool PR, int RPS, int NSG, int RB>
METAL_FUNC void rows_qmv_norm_sx(
    const device uint32_t* w,
    const device bfloat16_t* scales,
    const device bfloat16_t* biases,
    const device float* mixed,
    const device float* part,
    const device bfloat16_t* ln_w,
    const int gi,
    const float eps_ln,
    device bfloat16_t* y,
    const int N,
    const uint ty,
    const uint b0,
    const uint simd_gid,
    const uint simd_lid,
    threadgroup bfloat16_t* sx_all,
    threadgroup float* sm_all) {
  constexpr int block_size = 256;
  constexpr bool FULL = (B % RB) == 0;
  const int out_row = int(ty) * (NSG * RPS) + int(simd_gid) * RPS;
  int last_row = RPS - 1;
  bool active = true;
  if (PR) {
    if (out_row >= N) active = false;
    else last_row = min(last_row, N - 1 - out_row);
  }
  const int nb = FULL ? RB : min(RB, B - int(b0));
  const device float* pp0 = part + (size_t(b0) * 3u + uint(gi)) * 36u;
  constexpr bool ONE_INV = (RB <= NSG);
  float inv_my = 0.0f;
  thread float inv_arr[ONE_INV ? 1 : RB];
  if constexpr (ONE_INV) {
    if (int(simd_gid) < nb) {
      inv_my = proj_inv_from_part_q(pp0 + size_t(simd_gid) * 108u, eps_ln, simd_lid);
    }
  } else {
    _Pragma("clang loop unroll(full)")
    for (int b = 0; b < RB; b++) {
      inv_arr[b] = (FULL || b < nb) ? proj_inv_from_part_q(pp0 + size_t(b) * 108u, eps_ln, simd_lid) : 0.0f;
    }
  }
  float result[RB][RPS];
  for (int b = 0; b < RB; b++)
    for (int r = 0; r < RPS; r++) result[b][r] = 0.0f;
  const device uint8_t* ws = (const device uint8_t*)w + size_t(active ? out_row : 0) * 2304u + simd_lid * 8u;
  scales += size_t(active ? out_row : 0) * 36u + (simd_lid >> 3);
  biases += size_t(active ? out_row : 0) * 36u + (simd_lid >> 3);
  const device float* mp0 = mixed + (size_t(b0) * 3u + uint(gi)) * 2304u;
  y += size_t(b0) * N + out_row;
  const uint tid = simd_gid * 32u + simd_lid;
  const uint nthr = uint(NSG) * 32u;

  for (int k = 0; k < K; k += block_size) {
    const uint buf = uint(k >> 8) & 1u;
    threadgroup bfloat16_t* sx = sx_all + buf * (RB * 256u);
    threadgroup float* sm_s = sm_all + buf * (RB * 32u);
    if constexpr (ONE_INV) {
      if (int(simd_gid) < nb) {
        const uint b = simd_gid;
        const uint l = simd_lid;
        const device float* mp = mp0 + size_t(b) * (3u * 2304u) + k + l * 8u;
        uint4 lwpk = ((const device uint4*)(ln_w + k + l * 8u))[0];
        bfloat4 l0 = as_type<bfloat4>(lwpk.xy), l1 = as_type<bfloat4>(lwpk.zw);
        float4 m0 = ((const device float4*)mp)[0], m1 = ((const device float4*)(mp + 4))[0];
        bfloat4 b0v = bfloat4(
            bfloat16_t(glue_rms_apply<float>(m0.x, inv_my, l0.x)),
            bfloat16_t(glue_rms_apply<float>(m0.y, inv_my, l0.y)),
            bfloat16_t(glue_rms_apply<float>(m0.z, inv_my, l0.z)),
            bfloat16_t(glue_rms_apply<float>(m0.w, inv_my, l0.w)));
        bfloat4 b1v = bfloat4(
            bfloat16_t(glue_rms_apply<float>(m1.x, inv_my, l1.x)),
            bfloat16_t(glue_rms_apply<float>(m1.y, inv_my, l1.y)),
            bfloat16_t(glue_rms_apply<float>(m1.z, inv_my, l1.z)),
            bfloat16_t(glue_rms_apply<float>(m1.w, inv_my, l1.w)));
        uint2 u0 = as_type<uint2>(b0v), u1 = as_type<uint2>(b1v);
        ((threadgroup uint4*)(sx + b * 256u + l * 8u))[0] = uint4(u0.x, u0.y, u1.x, u1.y);
        float s = 0.0f;
        s += float(b0v.x); s += float(b0v.y); s += float(b0v.z); s += float(b0v.w);
        s += float(b1v.x); s += float(b1v.y); s += float(b1v.z); s += float(b1v.w);
        sm_s[b * 32u + l] = s;
      }
    } else {
      for (uint idx = tid; idx < uint(nb * 32); idx += nthr) {
        uint b = idx >> 5;
        uint l = idx & 31u;
        const device float* mp = mp0 + size_t(b) * (3u * 2304u) + k + l * 8u;
        uint4 lwpk = ((const device uint4*)(ln_w + k + l * 8u))[0];
        bfloat4 l0 = as_type<bfloat4>(lwpk.xy), l1 = as_type<bfloat4>(lwpk.zw);
        float4 m0 = ((const device float4*)mp)[0], m1 = ((const device float4*)(mp + 4))[0];
        float iv = inv_arr[b];
        bfloat4 b0v = bfloat4(
            bfloat16_t(glue_rms_apply<float>(m0.x, iv, l0.x)),
            bfloat16_t(glue_rms_apply<float>(m0.y, iv, l0.y)),
            bfloat16_t(glue_rms_apply<float>(m0.z, iv, l0.z)),
            bfloat16_t(glue_rms_apply<float>(m0.w, iv, l0.w)));
        bfloat4 b1v = bfloat4(
            bfloat16_t(glue_rms_apply<float>(m1.x, iv, l1.x)),
            bfloat16_t(glue_rms_apply<float>(m1.y, iv, l1.y)),
            bfloat16_t(glue_rms_apply<float>(m1.z, iv, l1.z)),
            bfloat16_t(glue_rms_apply<float>(m1.w, iv, l1.w)));
        uint2 u0 = as_type<uint2>(b0v), u1 = as_type<uint2>(b1v);
        ((threadgroup uint4*)(sx + b * 256u + l * 8u))[0] = uint4(u0.x, u0.y, u1.x, u1.y);
        float s = 0.0f;
        s += float(b0v.x); s += float(b0v.y); s += float(b0v.z); s += float(b0v.w);
        s += float(b1v.x); s += float(b1v.y); s += float(b1v.z); s += float(b1v.w);
        sm_s[b * 32u + l] = s;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (active) {
      thread uint2 wr[RPS];
      thread float sc[RPS], bi[RPS];
      _Pragma("clang loop unroll(full)")
      for (int row = 0; row < RPS; row++) {
        int src = PR ? min(row, last_row) : row;
        wr[row] = ((const device uint2*)(ws + src * 2304))[0];
        sc[row] = float(scales[src * 36]);
        bi[row] = float(biases[src * 36]);
      }
      _Pragma("clang loop unroll(full)")
      for (int row = 0; row < RPS; row++) {
        float4 w0 = float4(as_type<uchar4>(wr[row].x));
        float4 w1 = float4(as_type<uchar4>(wr[row].y));
        float scr = sc[row], bir = bi[row];
        _Pragma("clang loop unroll(full)")
        for (int b = 0; b < RB; b++) {
          if (FULL || b < nb) {
            uint4 xpk = ((const threadgroup uint4*)(sx + b * 256u + simd_lid * 8u))[0];
            bfloat4 x0 = as_type<bfloat4>(xpk.xy), x1 = as_type<bfloat4>(xpk.zw);
            float smb = sm_s[b * 32u + simd_lid];
            float accum = 0.0f;
            accum += float(x0.x) * w0.x; accum += float(x0.y) * w0.y; accum += float(x0.z) * w0.z; accum += float(x0.w) * w0.w;
            accum += float(x1.x) * w1.x; accum += float(x1.y) * w1.y; accum += float(x1.z) * w1.z; accum += float(x1.w) * w1.w;
            result[b][row] += scr * accum + smb * bir;
          }
        }
      }
    }
    ws += 256;
    scales += 4;
    biases += 4;
  }
  if (active) {
    _Pragma("clang loop unroll(full)")
    for (int b = 0; b < RB; b++) {
      _Pragma("clang loop unroll(full)")
      for (int row = 0; row < RPS; row++) {
        float r = simd_sum(result[b][row]);
        if (simd_lid == 0 && row <= last_row && (FULL || b < nb)) {
          y[size_t(b) * N + row] = static_cast<bfloat16_t>(r);
        }
      }
    }
  }
}
"""

_qnorm_header = None
_qnorm_kernels = {}
_QNORM_SX = os.environ.get("LOOPLING_QNORM_SX", "1") not in ("0", "")


def _get_qnorm_kernel(shared_ln, eps_ln, use_tile, use_sx=False):
    global _qnorm_header
    key = (bool(shared_ln), eps_ln, bool(use_tile), bool(use_sx))
    k = _qnorm_kernels.get(key)
    if k is not None:
        return k
    eps_lit = flit(eps_ln)
    if _qnorm_header is None:
        _qnorm_header = _qmv.base_header() + GLUE_HEADER + _QNORM_SRC
    ins = ["mixed", "part"] + (["lw0"] if shared_ln else ["lw0", "lw1", "lw2"])
    for i in range(3):
        ins += [f"w{i}", f"s{i}", f"b{i}"]
    lines = []
    if use_sx:
        lines += [
            "threadgroup uint4 sx_raw[2 * RB * 32];",
            "threadgroup bfloat16_t* sx_all = (threadgroup bfloat16_t*)sx_raw;",
            "threadgroup float sm_all[2 * RB * 32];",
        ]
    lines += [
        "uint ty = threadgroup_position_in_grid.y;",
        "uint rb0 = threadgroup_position_in_grid.x * RB;",
        "uint sg = simdgroup_index_in_threadgroup;",
        "uint sl = thread_index_in_simdgroup;",
    ]
    prev = "0"
    ut = "true" if use_tile else "false"
    for i in range(3):
        end = f"{prev} + ((N{i} + RPS * NSG - 1) / (RPS * NSG))"
        head = "if (ty < " + end + ")" if i == 0 else ("else if (ty < " + end + ")" if i == 1 else "else")
        yoff = "+".join(f"N{j}" for j in range(i)) or "0"
        lw_name = "lw0" if shared_ln else f"lw{i}"
        if use_sx:
            lines.append(
                f"{head} {{ rows_qmv_norm_sx<B, 2304, (N{i} % (RPS * NSG)) != 0, RPS, NSG, RB>"
                f"(w{i}, s{i}, b{i}, mixed, part, {lw_name}, {i}, {eps_lit}, y + B * ({yoff}), N{i}, ty - ({prev}), rb0, sg, sl, sx_all, sm_all); }}"
            )
        else:
            lines.append(
                f"{head} {{ rows_qmv_norm_run<B, 2304, (N{i} % (RPS * NSG)) != 0, RPS, NSG, RB, {ut}>"
                f"(w{i}, s{i}, b{i}, mixed, part, {lw_name}, {i}, {eps_lit}, y + B * ({yoff}), N{i}, ty - ({prev}), rb0, sg, sl); }}"
            )
        prev = end
    tag = "sx" if use_sx else ("t" if use_tile else "r")
    k = _qnorm_kernels[key] = mx.fast.metal_kernel(
        name="f3_qmv_norm_%s_%s_%s" % ("s" if shared_ln else "f", tag, eps_lit.replace(".", "p").replace("-", "m")),
        input_names=ins,
        output_names=["y"],
        source="\n".join(lines),
        header=_qnorm_header,
        ensure_row_contiguous=True,
    )
    return k


def qkv_rows_from_mixed(a, mixed, part, ln_ws, *, eps_ln=1e-6, full=None, flat=False):
    """Fused K3b+K4: reads K3's `(mixed (B,3,2304) f32, part (B,3,36) f32)` and normalizes on load inside K4 QKV."""
    is_dict = isinstance(a, dict)
    if full is None:
        ks = a["k_sconv"] if is_dict else a.k_sconv
        full = ("weight_proj" in ks) if isinstance(ks, dict) else hasattr(ks, "weight_proj")
    cat = _get_qkv_cat(a, False, full)
    if cat is None or mixed.ndim != 3 or mixed.shape[1:] != (3, 2304) or mixed.dtype != mx.float32:
        raise ValueError("qkv_rows_from_mixed requires cat aff8 QKV and mixed (B,3,2304) f32")
    B = mixed.shape[0]
    if not 1 <= B <= MAX_B or part.shape != (B, 3, 36) or part.dtype != mx.float32:
        raise ValueError("part must be (B,3,36) f32")
    cat_groups, grp_widths, fields, flat_w, b_cache = cat
    shared_ln = len(ln_ws) == 1 or (ln_ws[0] is ln_ws[1] and ln_ws[1] is ln_ws[2])
    mkey = ("qn", B, shared_ln, eps_ln)
    meta = b_cache.get(mkey)
    if meta is None:
        use_sx = _QNORM_SX and B >= 2
        use_tile = B >= 4
        if B == 1:
            rps, nsg, rb = 4, 2, 1
        elif use_sx:
            rps, nsg = 4, 4
            rb = B if B <= 4 else (3 if (B % 3 == 0 and B % 4 != 0) else 4)
        elif B <= 3:
            rps, nsg, rb = 2, 4, B
        else:
            rps, nsg, rb = (4, 2, 4) if B == 4 else (2, 4, min(B, 8))
        tile_w = rps * nsg
        tmpl = [("B", B), ("RPS", rps), ("NSG", nsg), ("RB", rb)]
        tiles = 0
        tot_n = 0
        for i, gw in enumerate(grp_widths):
            tmpl.append((f"N{i}", gw))
            tiles += (gw + tile_w - 1) // tile_w
            tot_n += gw
        grid = (((B + rb - 1) // rb) * 32, nsg * tiles, 1)
        tg = (32, nsg, 1)
        out_shp = [(B * tot_n,)]
        out_dt = [mx.bfloat16]
        kern = _get_qnorm_kernel(shared_ln, eps_ln, use_tile, use_sx=use_sx)
        grp_bases = (0, B * grp_widths[0], B * (grp_widths[0] + grp_widths[1]))
        lay = [(nm, grp_bases[gi] + loc, gw, nc) for nm, gi, loc, gw, nc in fields]
        meta = b_cache[mkey] = (kern, tmpl, grid, tg, out_shp, out_dt, lay, grp_bases)
    kern, tmpl, grid, tg, out_shp, out_dt, lay, grp_bases = meta
    args = ([mixed, part, ln_ws[0]] + flat_w) if shared_ln else ([mixed, part, ln_ws[0], ln_ws[1], ln_ws[2]] + flat_w)
    buf = kern(
        inputs=args,
        template=tmpl,
        grid=grid,
        threadgroup=tg,
        output_shapes=out_shp,
        output_dtypes=out_dt,
    )[0]
    if flat:
        return buf, lay
    out = {}
    for nm, gi, loc, gw, nc in fields:
        gb = grp_bases[gi]
        view = buf[gb : gb + B * gw].reshape(B, gw)[:, loc : loc + nc]
        out[nm] = view if (loc == 0 and gw == nc) else mx.contiguous(view)
    return out


def qkv_msg_rows(layer, r1, msg, *, flat=True):
    """Fused K2 -> K3(with partials) -> K3b+K4 QKV for a message visit (eliminates QKV K3b launch)."""
    is_dict = isinstance(layer, dict)
    a = layer["self_attn"] if is_dict else layer.self_attn
    sliding = ("input_layernorm" in layer) if is_dict else hasattr(layer, "input_layernorm")
    gs = (
        (layer["q_message_gate"], layer["k_message_gate"], layer["v_message_gate"])
        if is_dict
        else (layer.q_message_gate, layer.k_message_gate, layer.v_message_gate)
    )
    down = gate_down_rows(r1["qkv_gate_in"], [g["down_proj"]["weight"] for g in gs])
    mixed, part = gate_mix_part_rows(
        down,
        [r1["s_q"], r1["s_k"], r1["s_v"]],
        msg,
        (0, 1, 2),
        [g["forget_gate_proj"]["weight"] for g in gs],
        [g["message_gate_proj"]["weight"] for g in gs],
    )
    if sliding:
        ln_m = layer["input_layernorm"] if is_dict else layer.input_layernorm
        ln_ws, eps_ln = [ln_m["weight"]], ln_m.eps
    else:
        m0 = layer["q_input_layernorm"] if is_dict else layer.q_input_layernorm
        m1 = layer["k_input_layernorm"] if is_dict else layer.k_input_layernorm
        m2 = layer["v_input_layernorm"] if is_dict else layer.v_input_layernorm
        ln_ws, eps_ln = [m0["weight"], m1["weight"], m2["weight"]], m0.eps
    return qkv_rows_from_mixed(a, mixed, part, ln_ws, eps_ln=eps_ln, full=not sliding, flat=flat)


def mlp_msg_part(layer, r10, msg):
    """Fused K2 -> K3(with partials) for mlp_message_gate -> (mixed (B,1,2304) f32, part (B,1,36) f32, ln_w, eps_ln)."""
    is_dict = isinstance(layer, dict)
    g = layer["mlp_message_gate"] if is_dict else layer.mlp_message_gate
    ln_m = layer["post_attention_layernorm"] if is_dict else layer.post_attention_layernorm
    s_mlp = r10["mlp_src"]
    B = int(s_mlp.shape[0])
    gin = r10["mlp_gate_in"].reshape(B, 1, 4608)
    m4 = msg.reshape(B, 4, 2304)
    down = gate_down_rows(gin, [g["down_proj"]["weight"]])
    mixed, part = gate_mix_part_rows(
        down, [s_mlp], m4, (3,),
        [g["forget_gate_proj"]["weight"]], [g["message_gate_proj"]["weight"]],
    )
    return mixed, part, ln_m["weight"], ln_m.eps


# =============================================================================================== K4+K5 fused (B=1 QKV+MID+REL)
_qkv_all_cat_cache: dict = {}


def _get_qkv_all_cat(a, full):
    """Single concatenated (W_all, S_all, B_all) for B=1 fused QKV+MID+REL.
    Row order: q_proj(3072), r_proj(384), gate(3072), sinks(24), k_proj(1024), v_proj(1024) [, k_wproj(32), v_wproj(32)]."""
    if not full:
        cat = _get_qkv_cat(a, True, False)
        if cat is not None:
            return cat[0][0]
    is_dict = isinstance(a, dict)
    if is_dict and not hasattr(a, "q_proj"):
        mods = [a["q_proj"], a["r_proj"], a["gate_proj"], a["sink_proj"], a["k_proj"], a["v_proj"]]
        if full:
            mods += [a["k_sconv"]["weight_proj"], a["v_sconv"]["weight_proj"]]
    else:
        mods = [a.q_proj, a.r_proj, a.gate_proj, a.sink_proj, a.k_proj, a.v_proj]
        if full:
            mods += [a.k_sconv.weight_proj, a.v_sconv.weight_proj]
    if not all(_aff8_ok(m) for m in mods):
        return None
    cur_refs = tuple(t for m in mods for t in (m["weight"], m["scales"], m["biases"]))
    key = (id(a), bool(full))
    ent = _qkv_all_cat_cache.get(key)
    if ent is not None and len(ent[0]) == len(cur_refs) and all(x is y for x, y in zip(ent[0], cur_refs)):
        return ent[1]
    if len(_qkv_all_cat_cache) >= 256:
        _qkv_all_cat_cache.clear()
    cw = mx.contiguous(mx.concatenate([m["weight"] for m in mods], axis=0))
    cs = mx.contiguous(mx.concatenate([m["scales"] for m in mods], axis=0))
    cb = mx.contiguous(mx.concatenate([m["biases"] for m in mods], axis=0))
    val = (cw, cs, cb)
    mx.eval(val)
    _qkv_all_cat_cache[key] = (cur_refs, val)
    if not is_dict or hasattr(a, "__weakref__"):
        try:
            weakref.finalize(a, _qkv_all_cat_cache.pop, key, None)
        except TypeError:
            pass
    return val


_QKV_MID_HEADER_EXTRA = r"""
template <bool FROM_TG>
inline void qmv_b1_4rows_impl(
    const device uint32_t* w,
    const device bfloat16_t* scales,
    const device bfloat16_t* biases,
    const threadgroup bfloat16_t* sx,
    const device bfloat16_t* dx,
    const int row0,
    const uint sl,
    thread float* out4) {
  const device uint8_t* ws = (const device uint8_t*)w + size_t(row0) * 2304u + sl * 8u;
  const device bfloat16_t* sp = scales + size_t(row0) * 36u + (sl >> 3);
  const device bfloat16_t* bp = biases + size_t(row0) * 36u + (sl >> 3);
  float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f;
  for (int k = 0; k < 2304; k += 256) {
    uint4 xpk = FROM_TG
        ? ((const threadgroup uint4*)(sx + k + sl * 8u))[0]
        : ((const device uint4*)(dx + k + sl * 8u))[0];
    uint2 wr0 = ((const device uint2*)ws)[0];
    uint2 wr1 = ((const device uint2*)(ws + 2304u))[0];
    uint2 wr2 = ((const device uint2*)(ws + 4608u))[0];
    uint2 wr3 = ((const device uint2*)(ws + 6912u))[0];
    bfloat16_t s0 = sp[0], s1 = sp[36], s2 = sp[72], s3 = sp[108];
    bfloat16_t b0 = bp[0], b1 = bp[36], b2 = bp[72], b3 = bp[108];
    bfloat4 x0b = as_type<bfloat4>(xpk.xy);
    bfloat4 x1b = as_type<bfloat4>(xpk.zw);
    float x0 = float(x0b.x), x1 = float(x0b.y), x2 = float(x0b.z), x3 = float(x0b.w);
    float x4 = float(x1b.x), x5 = float(x1b.y), x6 = float(x1b.z), x7 = float(x1b.w);
    float smb = 0.0f;
    smb += x0; smb += x1; smb += x2; smb += x3;
    smb += x4; smb += x5; smb += x6; smb += x7;
    {
      float4 w0 = float4(as_type<uchar4>(wr0.x)), w1 = float4(as_type<uchar4>(wr0.y));
      float a = 0.0f;
      a += x0 * w0.x; a += x1 * w0.y; a += x2 * w0.z; a += x3 * w0.w;
      a += x4 * w1.x; a += x5 * w1.y; a += x6 * w1.z; a += x7 * w1.w;
      r0 += float(s0) * a + smb * float(b0);
    }
    {
      float4 w0 = float4(as_type<uchar4>(wr1.x)), w1 = float4(as_type<uchar4>(wr1.y));
      float a = 0.0f;
      a += x0 * w0.x; a += x1 * w0.y; a += x2 * w0.z; a += x3 * w0.w;
      a += x4 * w1.x; a += x5 * w1.y; a += x6 * w1.z; a += x7 * w1.w;
      r1 += float(s1) * a + smb * float(b1);
    }
    {
      float4 w0 = float4(as_type<uchar4>(wr2.x)), w1 = float4(as_type<uchar4>(wr2.y));
      float a = 0.0f;
      a += x0 * w0.x; a += x1 * w0.y; a += x2 * w0.z; a += x3 * w0.w;
      a += x4 * w1.x; a += x5 * w1.y; a += x6 * w1.z; a += x7 * w1.w;
      r2 += float(s2) * a + smb * float(b2);
    }
    {
      float4 w0 = float4(as_type<uchar4>(wr3.x)), w1 = float4(as_type<uchar4>(wr3.y));
      float a = 0.0f;
      a += x0 * w0.x; a += x1 * w0.y; a += x2 * w0.z; a += x3 * w0.w;
      a += x4 * w1.x; a += x5 * w1.y; a += x6 * w1.z; a += x7 * w1.w;
      r3 += float(s3) * a + smb * float(b3);
    }
    ws += 256u;
    sp += 4u;
    bp += 4u;
  }
  out4[0] = simd_sum(r0);
  out4[1] = simd_sum(r1);
  out4[2] = simd_sum(r2);
  out4[3] = simd_sum(r3);
}
"""

_qkv_mid_header = None
_qkv_mid_kernels = {}
_qkv_mid_layer_cache: dict = {}
_QKVMID_NSG_DEF = int(os.environ.get("LOOPLING_QKVMID_NSG", "16"))
if _QKVMID_NSG_DEF not in (8, 16, 32):
    _QKVMID_NSG_DEF = 16


def _qkv_mid_source(full, from_mixed, single_x, shared_ln, thr_q, thr_k, thr_r, eps_v, eps_ln):
    scale_conv = flit(np.float32(math.sqrt(5)))
    if from_mixed:
        lw_expr = "lw0" if shared_ln else "((gi == 0) ? lw0 : ((gi == 1) ? lw1 : lw2))"
        tg_decls = """
    threadgroup uint4 sx_raw[288];
    threadgroup bfloat16_t* sx = (threadgroup bfloat16_t*)sx_raw;
    threadgroup float* st_buf = (threadgroup float*)sx_raw;"""
        load_x = f"""
    const float inv = proj_inv_from_part_q(part + uint(gi) * 36u, {flit(eps_ln)}, sl);
    const device float* mp = mixed + size_t(gi) * 2304u;
    const device bfloat16_t* lw = {lw_expr};
    for (uint idx = tid; idx < 576u; idx += uint(NSG) * 32u) {{
      float4 mv = ((const device float4*)mp)[idx];
      bfloat4 lv = as_type<bfloat4>(((const device uint2*)lw)[idx]);
      bfloat4 ov = bfloat4(
          bfloat16_t(glue_rms_apply<float>(mv.x, inv, lv.x)),
          bfloat16_t(glue_rms_apply<float>(mv.y, inv, lv.y)),
          bfloat16_t(glue_rms_apply<float>(mv.z, inv, lv.z)),
          bfloat16_t(glue_rms_apply<float>(mv.w, inv, lv.w)));
      ((threadgroup uint2*)sx)[idx] = as_type<uint2>(ov);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const device bfloat16_t* dx = (const device bfloat16_t*)0;"""
        qmv_call = "qmv_b1_4rows_impl<true>(w_all, s_all, b_all, sx, dx, "
    else:
        tg_decls = """
    threadgroup float st_buf[8 * 64];
    const threadgroup bfloat16_t* sx = (const threadgroup bfloat16_t*)0;"""
        load_x = (
            "    const device bfloat16_t* dx = x0;"
            if single_x
            else "    const device bfloat16_t* dx = (gi == 0) ? x0 : ((gi == 1) ? x1 : x2);"
        )
        qmv_call = "qmv_b1_4rows_impl<false>(w_all, s_all, b_all, sx, dx, "

    k_wproj_block = f"""
      if (sg == 0u) {{
        float wpr[4];
        {qmv_call}8600 + int(hk) * 4, sl, wpr);
        if (sl == 0u) {{
          for (int r = 0; r < 4; r++) wp_s[r] = float(bfloat16_t(wpr[r]));
        }}
      }}""" if full else ""
    v_wproj_block = f"""
      if (sg == 0u) {{
        float wpr[4];
        {qmv_call}8632 + int(hk) * 4, sl, wpr);
        if (sl == 0u) {{
          for (int r = 0; r < 4; r++) wp_s[r] = float(bfloat16_t(wpr[r]));
        }}
      }}""" if full else ""
    k_conv_expr = """
          float sw[4];
          for (int k = 0; k < 4; k++) sw[k] = float(k_static[k * 1024u + c]);
          y[i] = mid_conv_d(x0v, s0v, s1v, s2v, sw, dy);""" if full else f"""
          const device bfloat16_t* cw = k_conv + c * 4u;
          y[i] = mid_conv_s(x0v, s0v, s1v, s2v, float(cw[0]), float(cw[1]), float(cw[2]), float(cw[3]), {scale_conv});"""
    v_conv_expr = """
          float sw[4];
          for (int k = 0; k < 4; k++) sw[k] = float(v_static[k * 1024u + c]);
          y[i] = mid_conv_d(x0v, s0v, s1v, s2v, sw, dy);""" if full else f"""
          const device bfloat16_t* cw = v_conv + c * 4u;
          y[i] = mid_conv_s(x0v, s0v, s1v, s2v, float(cw[0]), float(cw[1]), float(cw[2]), float(cw[3]), {scale_conv});"""

    return f"""
    constexpr int PASSES = 128 / (NSG * 4);
    constexpr uint GATE_ROWS = uint(NSG * 4);
    constexpr uint N_GATE_TG = 3072u / GATE_ROWS;{tg_decls}
    threadgroup float head_f[128];
    threadgroup bfloat16_t rel_s[128];
    {"threadgroup float wp_s[4];" if full else ""}
    const uint tg = threadgroup_position_in_grid.x;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint sl = thread_index_in_simdgroup;
    const uint tid = sg * 32u + sl;
    const int gi = (tg < 24u) ? 0 : ((tg < 32u) ? 1 : ((tg < 40u) ? 2 : 0));
{load_x}
    {"const float tau_v = tau[0];" if full else ""}

    if (tg < 24u) {{
      const uint h = tg;
      const int base_row = int(h) * 128;
      for (int p = 0; p < PASSES; p++) {{
        int lrow = p * (NSG * 4) + int(sg) * 4;
        float out4[4];
        {qmv_call}base_row + lrow, sl, out4);
        if (sl == 0u) {{
          for (int r = 0; r < 4; r++) head_f[lrow + r] = float(bfloat16_t(out4[r]));
        }}
      }}
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sg == 0u) {{
        float s[4];
        const uint qoff = h * 128u + sl * 4u;
        for (int i = 0; i < 4; i++) s[i] = head_f[sl * 4u + i] * float(q_scale[qoff + i]);
        glue_l2_clip_lane128(s, {flit(thr_q)});
        bfloat16_t qo[4];
        for (int i = 0; i < 4; i++) {{
          bfloat16_t o = bfloat16_t(s[i]);
          {"o = bfloat16_t(glue_tau(float(o), tau_v));" if full else ""}
          qo[i] = o;
        }}
        bfloat4 ob = bfloat4(qo[0], qo[1], qo[2], qo[3]);
        ((device uint2*)(q_out + qoff))[0] = as_type<uint2>(ob);
      }}
    }} else if (tg < 32u) {{
      const uint hk = tg - 24u;
      const int base_row = 6552 + int(hk) * 128;
      for (int p = 0; p < PASSES; p++) {{
        int lrow = p * (NSG * 4) + int(sg) * 4;
        float out4[4];
        {qmv_call}base_row + lrow, sl, out4);
        if (sl == 0u) {{
          for (int r = 0; r < 4; r++) head_f[lrow + r] = float(bfloat16_t(out4[r]));
        }}
      }}{k_wproj_block}
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sg == 0u) {{
        {"float dy[4] = {wp_s[0], wp_s[1], wp_s[2], wp_s[3]};" if full else ""}
        const uint c0 = hk * 128u + sl * 4u;
        float y[4];
        for (int i = 0; i < 4; i++) {{
          uint c = c0 + i;
          float x0v = head_f[sl * 4u + i];
          float s0v = cs0[c], s1v = cs0[1024u + c], s2v = cs0[2048u + c];
          {k_conv_expr}
          w0[c] = s1v; w0[1024u + c] = s2v; w0[2048u + c] = x0v;
        }}
        glue_l2_clip_lane128(y, {flit(thr_k)});
        bfloat4 ob = bfloat4(bfloat16_t(y[0]), bfloat16_t(y[1]), bfloat16_t(y[2]), bfloat16_t(y[3]));
        ((device uint2*)(kv_out + c0))[0] = as_type<uint2>(ob);
      }}
    }} else if (tg < 40u) {{
      const uint hk = tg - 32u;
      const int base_row = 7576 + int(hk) * 128;
      for (int p = 0; p < PASSES; p++) {{
        int lrow = p * (NSG * 4) + int(sg) * 4;
        float out4[4];
        {qmv_call}base_row + lrow, sl, out4);
        if (sl == 0u) {{
          for (int r = 0; r < 4; r++) head_f[lrow + r] = float(bfloat16_t(out4[r]));
        }}
      }}{v_wproj_block}
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sg == 0u) {{
        {"float dy[4] = {wp_s[0], wp_s[1], wp_s[2], wp_s[3]};" if full else ""}
        const uint c0 = hk * 128u + sl * 4u;
        float y[4];
        for (int i = 0; i < 4; i++) {{
          uint c = c0 + i;
          float x0v = head_f[sl * 4u + i];
          float s0v = cs1[c], s1v = cs1[1024u + c], s2v = cs1[2048u + c];
          {v_conv_expr}
          w1[c] = s1v; w1[1024u + c] = s2v; w1[2048u + c] = x0v;
        }}
        for (int i = 0; i < 4; i++) head_f[sl * 4u + i] = y[i];
        float inv_v = glue_rms_inv_sg<128, true>(head_f, {flit(eps_v)}, sl);
        bfloat4 ob = bfloat4(
            glue_rms_apply_nw<bfloat16_t>(y[0], inv_v),
            glue_rms_apply_nw<bfloat16_t>(y[1], inv_v),
            glue_rms_apply_nw<bfloat16_t>(y[2], inv_v),
            glue_rms_apply_nw<bfloat16_t>(y[3], inv_v));
        ((device uint2*)(kv_out + 1024u + c0))[0] = as_type<uint2>(ob);
      }}
    }} else if (tg < 43u) {{
      const uint rg = tg - 40u;
      const int base_row = 3072 + int(rg) * 128;
      for (int p = 0; p < PASSES; p++) {{
        int lrow = p * (NSG * 4) + int(sg) * 4;
        float out4[4];
        {qmv_call}base_row + lrow, sl, out4);
        if (sl == 0u) {{
          for (int r = 0; r < 4; r++) head_f[lrow + r] = float(bfloat16_t(out4[r]));
        }}
      }}
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sg == 0u && sl < 8u) {{
        const uint h = rg * 8u + sl;
        float s[16];
        for (int i = 0; i < 16; i++) s[i] = head_f[sl * 16u + i] * float(rel_scale[h * 16u + i]);
        glue_l2_clip_row16(s, {flit(thr_r)});
        for (int i = 0; i < 16; i++) {{
          bfloat16_t o = bfloat16_t(s[i]);
          {"o = bfloat16_t(glue_tau(float(o), tau_v));" if full else ""}
          rel_s[sl * 16u + i] = o;
        }}
      }}
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sg < 8u) {{
        simdgroup_bfloat8x8 ma0, ma1;
        simdgroup_load(ma0, rel_s + 0, 16);
        simdgroup_load(ma1, rel_s + 8, 16);
        threadgroup float* st = st_buf + sg * 64u;
        for (uint n0 = sg * 16u; n0 < uint(REL_R); n0 += 128u) {{
          const device bfloat16_t* Bm = rw + n0;
          device bfloat16_t* Cm = pb_out + size_t(rg * 8u) * uint(REL_R) + n0;
          for (int ni = 0; ni < 2; ni++) {{
            simdgroup_bfloat8x8 mb0, mb1;
            simdgroup_load(mb0, Bm + 0 * REL_R + ni * 8, REL_R);
            simdgroup_load(mb1, Bm + 8 * REL_R + ni * 8, REL_R);
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            simdgroup_multiply_accumulate(acc, ma0, mb0, acc);
            simdgroup_multiply_accumulate(acc, ma1, mb1, acc);
            simdgroup_store(acc, st, 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (int idx = int(sl); idx < 64; idx += 32) {{
              int r = idx >> 3, c = idx & 7;
              Cm[r * REL_R + ni * 8 + c] = bfloat16_t(st[idx]);
            }}
            simdgroup_barrier(mem_flags::mem_threadgroup);
          }}
        }}
      }}
    }} else if (tg < 43u + N_GATE_TG) {{
      const uint gg = tg - 43u;
      const int g_off = int(gg * GATE_ROWS) + int(sg) * 4;
      float out4[4];
      {qmv_call}3456 + g_off, sl, out4);
      if (sl == 0u) {{
        bfloat4 ob = bfloat4(bfloat16_t(out4[0]), bfloat16_t(out4[1]), bfloat16_t(out4[2]), bfloat16_t(out4[3]));
        ((device uint2*)(gs_out + g_off))[0] = as_type<uint2>(ob);
      }}
    }} else {{
      if (sg < 6u) {{
        int lrow = int(sg) * 4;
        float out4[4];
        {qmv_call}6528 + lrow, sl, out4);
        if (sl == 0u) {{
          bfloat4 ob = bfloat4(bfloat16_t(out4[0]), bfloat16_t(out4[1]), bfloat16_t(out4[2]), bfloat16_t(out4[3]));
          ((device uint2*)(gs_out + 3072 + lrow))[0] = as_type<uint2>(ob);
        }}
      }}
    }}
"""


def _get_qkv_mid_kernel(full, from_mixed, single_x, shared_ln, thr_q, thr_k, thr_r, eps_v, eps_ln, dtypes):
    global _qkv_mid_header
    key = (bool(full), bool(from_mixed), bool(single_x), bool(shared_ln), thr_q, thr_k, thr_r, eps_v, eps_ln, dtypes)
    k = _qkv_mid_kernels.get(key)
    if k is not None:
        return k
    if _qkv_mid_header is None:
        _qkv_mid_header = _qmv.base_header() + GLUE_HEADER + _mid._MID_HEADER + _QNORM_SRC + _QKV_MID_HEADER_EXTRA
    if from_mixed:
        ins = ["mixed", "part"] + (["lw0"] if shared_ln else ["lw0", "lw1", "lw2"])
    elif single_x:
        ins = ["x0"]
    else:
        ins = ["x0", "x1", "x2"]
    ins += ["w_all", "s_all", "b_all", "cs0", "cs1", "q_scale", "rel_scale", "rw"]
    if full:
        ins += ["tau", "k_static", "v_static"]
    else:
        ins += ["k_conv", "v_conv"]
    src = _qkv_mid_source(full, from_mixed, single_x, shared_ln, thr_q, thr_k, thr_r, eps_v, eps_ln)
    k = _qkv_mid_kernels[key] = mx.fast.metal_kernel(
        name="f3_qkvmid_%s%s%s_%d" % ("F" if full else "S", "m" if from_mixed else ("1" if single_x else "3"), "s" if shared_ln else "f", abs(hash(key)) % 10**9),
        input_names=ins,
        output_names=["q_out", "kv_out", "pb_out", "gs_out", "w0", "w1"],
        source=src,
        header=_qkv_mid_header,
        ensure_row_contiguous=True,
    )
    return k


def _get_qkv_mid_layer_meta(layer):
    is_dict = isinstance(layer, dict)
    a = layer["self_attn"] if is_dict else layer.self_attn
    a_dict = isinstance(a, dict)
    ql2 = a["q_l2_clip"] if a_dict else a.q_l2_clip
    kl2 = a["k_l2_clip"] if a_dict else a.k_l2_clip
    rl2 = a["rel_l2_clip"] if a_dict else a.rel_l2_clip
    q_sc = ql2["scale"] if isinstance(ql2, dict) else ql2.scale
    r_sc = rl2["scale"] if isinstance(rl2, dict) else rl2.scale
    ksconv = a["k_sconv"] if a_dict else a.k_sconv
    vsconv = a["v_sconv"] if a_dict else a.v_sconv
    full = a.sliding_window == 0
    assert a.num_heads == 24 and a.num_key_value_heads == 8 and a.head_dim == 128
    cat = _get_qkv_all_cat(a, full)
    if cat is None:
        raise ValueError("qkv_mid_rel_b1 requires aff8 gs64 QKV weights")
    if full:
        cw0 = ksconv["static_weight"] if isinstance(ksconv, dict) else ksconv.static_weight
        cw1 = vsconv["static_weight"] if isinstance(vsconv, dict) else vsconv.static_weight
        m0 = layer["q_input_layernorm"] if is_dict else layer.q_input_layernorm
        m1 = layer["k_input_layernorm"] if is_dict else layer.k_input_layernorm
        m2 = layer["v_input_layernorm"] if is_dict else layer.v_input_layernorm
        ln_ws, eps_ln = (m0["weight"], m1["weight"], m2["weight"]), m0.eps
    else:
        cw0 = ksconv["conv1d"]["weight"] if isinstance(ksconv, dict) else ksconv.conv1d.weight
        cw1 = vsconv["conv1d"]["weight"] if isinstance(vsconv, dict) else vsconv.conv1d.weight
        ln_m = layer["input_layernorm"] if is_dict else layer.input_layernorm
        ln_ws, eps_ln = (ln_m["weight"],), ln_m.eps
    has_msg = ("q_message_gate" in layer) if is_dict else hasattr(layer, "q_message_gate")
    if has_msg:
        gs = (
            (layer["q_message_gate"], layer["k_message_gate"], layer["v_message_gate"])
            if is_dict
            else (layer.q_message_gate, layer.k_message_gate, layer.v_message_gate)
        )
        g_down = [g["down_proj"]["weight"] for g in gs]
        g_wf = [g["forget_gate_proj"]["weight"] for g in gs]
        g_wm = [g["message_gate_proj"]["weight"] for g in gs]
    else:
        g_down = g_wf = g_wm = None
    return (
        full, cat[0], cat[1], cat[2], q_sc, r_sc, cw0, cw1,
        ql2.threshold, kl2.threshold, rl2.threshold, a.eps,
        ln_ws, eps_ln, g_down, g_wf, g_wm,
    )


def _run_qkv_mid_rel_b1(layer, cs0, cs1, rel_w, *, q_in=None, k_in=None, v_in=None, mixed=None, part=None, tau=None):
    (
        full, w_all, s_all, b_all, q_sc, r_sc, cw0, cw1,
        thr_q, thr_k, thr_r, eps_v, ln_ws, eps_ln, _, _, _,
    ) = _get_qkv_mid_layer_meta(layer)
    rel_r = int(rel_w.shape[1])
    from_mixed = mixed is not None
    if from_mixed:
        single_x = False
        shared_ln = len(ln_ws) == 1
        ins = [mixed, part] + list(ln_ws)
    else:
        shared_ln = True
        single_x = (q_in is k_in) and (k_in is v_in)
        ins = [q_in] if single_x else [q_in, k_in, v_in]
    ins += [w_all, s_all, b_all, cs0, cs1, q_sc, r_sc, rel_w]
    if full:
        ins += [tau.reshape(-1), cw0, cw1]
    else:
        ins += [cw0, cw1]
    nsg = _QKVMID_NSG_DEF
    n_tg = 44 + 3072 // (nsg * 4)
    kern = _get_qkv_mid_kernel(
        full, from_mixed, single_x, shared_ln,
        thr_q, thr_k, thr_r, eps_v, eps_ln,
        tuple(x.dtype for x in ins),
    )
    return kern(
        inputs=ins,
        template=[("NSG", nsg), ("REL_R", rel_r)],
        grid=(n_tg * nsg * 32, 1, 1),
        threadgroup=(nsg * 32, 1, 1),
        output_shapes=[(1, 24, 128), (2, 8, 1, 128), (1, 24, rel_r), (3096,), (1, 3, 1024), (1, 3, 1024)],
        output_dtypes=[mx.bfloat16, mx.bfloat16, mx.bfloat16, mx.bfloat16, mx.float32, mx.float32],
    )


def qkv_mid_rel_b1(layer, q_in, k_in, v_in, cs0, cs1, rel_w, *, tau=None):
    """Fused B=1 K4 QKV + K5 MID + REL MMA -> (q, kv_new, pb, gs_flat, w0, w1).
    gs_flat has gate at offset 0 (3072) and sinks at offset 3072 (24)."""
    return _run_qkv_mid_rel_b1(layer, cs0, cs1, rel_w, q_in=q_in, k_in=k_in, v_in=v_in, tau=tau)


def qkv_mid_rel_msg_b1(layer, r1, msg, cs0, cs1, rel_w, *, tau=None):
    """Fused B=1 K2 -> K3(part) -> (K3b + K4 QKV + K5 MID + REL MMA) -> (q, kv_new, pb, gs_flat, w0, w1)."""
    meta = _get_qkv_mid_layer_meta(layer)
    g_down, g_wf, g_wm = meta[14], meta[15], meta[16]
    down = gate_down_rows(r1["qkv_gate_in"], g_down)
    mixed, part = gate_mix_part_rows(down, [r1["s_q"], r1["s_k"], r1["s_v"]], msg, (0, 1, 2), g_wf, g_wm)
    return _run_qkv_mid_rel_b1(layer, cs0, cs1, rel_w, mixed=mixed, part=part, tau=tau)


_OPROJ_GSX_SRC = r"""
  constexpr int block_size = 256;
  constexpr bool FULL = (B % RB) == 0;
  uint ty = threadgroup_position_in_grid.y;
  uint b0 = threadgroup_position_in_grid.x * RB;
  uint simd_gid = simdgroup_index_in_threadgroup;
  uint simd_lid = thread_index_in_simdgroup;
  const int nb = FULL ? RB : min(RB, B - int(b0));
  const int out_row = int(ty) * (NSG * RPS) + int(simd_gid) * RPS;
  bool active = true;
  int last_row = RPS - 1;
  if (PR) {
    if (out_row >= N) active = false;
    else last_row = min(last_row, N - 1 - out_row);
  }
  const int in_vec_size_w = K;
  const int in_vec_size_g = K / 64;
  const device uint8_t* ws = (const device uint8_t*)w + size_t(active ? out_row : 0) * in_vec_size_w + simd_lid * 8;
  const device bfloat16_t* sc_p = scales + size_t(active ? out_row : 0) * in_vec_size_g + (simd_lid >> 3);
  const device bfloat16_t* bi_p = biases + size_t(active ? out_row : 0) * in_vec_size_g + (simd_lid >> 3);
  const device bfloat16_t* ap = a + size_t(b0) * K + simd_lid * 8;
  const device bfloat16_t* gp = g + size_t(b0) * K + simd_lid * 8;
  device bfloat16_t* yp = y + size_t(b0) * N + out_row;

  thread float result[RB][RPS];
  for (int b = 0; b < RB; b++)
    for (int r = 0; r < RPS; r++) result[b][r] = 0.0f;

  threadgroup uint4 sx_bf[(DBUF ? 2 : 1) * RB * 32];
  threadgroup float sm_f[(DBUF ? 2 : 1) * RB * 32];

  for (int k = 0; k < K; k += block_size) {
    const uint buf = DBUF ? (uint(k >> 8) & 1u) : 0u;
    threadgroup uint4* sx = sx_bf + buf * (RB * 32u);
    threadgroup float* sm = sm_f + buf * (RB * 32u);
    for (int b = int(simd_gid); b < nb; b += NSG) {
      uint4 apk = ((const device uint4*)(ap + size_t(b) * K))[0];
      uint4 gpk = ((const device uint4*)(gp + size_t(b) * K))[0];
      bfloat4 a0 = as_type<bfloat4>(apk.xy), a1 = as_type<bfloat4>(apk.zw);
      bfloat4 g0 = as_type<bfloat4>(gpk.xy), g1 = as_type<bfloat4>(gpk.zw);
      bfloat4 v0 = bfloat4(
          bfloat16_t(glue_gate(float(a0.x), float(g0.x))),
          bfloat16_t(glue_gate(float(a0.y), float(g0.y))),
          bfloat16_t(glue_gate(float(a0.z), float(g0.z))),
          bfloat16_t(glue_gate(float(a0.w), float(g0.w))));
      bfloat4 v1 = bfloat4(
          bfloat16_t(glue_gate(float(a1.x), float(g1.x))),
          bfloat16_t(glue_gate(float(a1.y), float(g1.y))),
          bfloat16_t(glue_gate(float(a1.z), float(g1.z))),
          bfloat16_t(glue_gate(float(a1.w), float(g1.w))));
      uint2 u0 = as_type<uint2>(v0), u1 = as_type<uint2>(v1);
      sx[b * 32u + simd_lid] = uint4(u0.x, u0.y, u1.x, u1.y);
      float s = 0.0f;
      s += float(v0.x); s += float(v0.y); s += float(v0.z); s += float(v0.w);
      s += float(v1.x); s += float(v1.y); s += float(v1.z); s += float(v1.w);
      sm[b * 32u + simd_lid] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (active) {
      if constexpr (PACKW) {
        thread uint2 wr_r[RPS];
        thread bfloat16_t sc_bf[RPS], bi_bf[RPS];
        _Pragma("clang loop unroll(full)")
        for (int r = 0; r < RPS; r++) {
          int src = PR ? min(r, last_row) : r;
          wr_r[r] = ((const device uint2*)(ws + size_t(src) * in_vec_size_w))[0];
          sc_bf[r] = sc_p[size_t(src) * in_vec_size_g];
          bi_bf[r] = bi_p[size_t(src) * in_vec_size_g];
        }
        _Pragma("clang loop unroll(full)")
        for (int b = 0; b < RB; b++) {
          if (FULL || b < nb) {
            uint4 xpk = sx[b * 32u + simd_lid];
            bfloat4 xv0 = as_type<bfloat4>(xpk.xy), xv1 = as_type<bfloat4>(xpk.zw);
            float x0 = float(xv0.x), x1 = float(xv0.y), x2 = float(xv0.z), x3 = float(xv0.w);
            float x4 = float(xv1.x), x5 = float(xv1.y), x6 = float(xv1.z), x7 = float(xv1.w);
            float smb = sm[b * 32u + simd_lid];
            _Pragma("clang loop unroll(full)")
            for (int r = 0; r < RPS; r++) {
              float4 w0 = float4(as_type<uchar4>(wr_r[r].x));
              float4 w1 = float4(as_type<uchar4>(wr_r[r].y));
              float acc = 0.0f;
              acc += x0 * w0.x; acc += x1 * w0.y; acc += x2 * w0.z; acc += x3 * w0.w;
              acc += x4 * w1.x; acc += x5 * w1.y; acc += x6 * w1.z; acc += x7 * w1.w;
              result[b][r] += float(sc_bf[r]) * acc + smb * float(bi_bf[r]);
            }
          }
        }
      } else {
        thread half4 w0[RPS], w1[RPS];
        thread float sc[RPS], bi[RPS];
        _Pragma("clang loop unroll(full)")
        for (int r = 0; r < RPS; r++) {
          int src = PR ? min(r, last_row) : r;
          uint2 wr = ((const device uint2*)(ws + size_t(src) * in_vec_size_w))[0];
          w0[r] = half4(as_type<uchar4>(wr.x));
          w1[r] = half4(as_type<uchar4>(wr.y));
          sc[r] = float(sc_p[size_t(src) * in_vec_size_g]);
          bi[r] = float(bi_p[size_t(src) * in_vec_size_g]);
        }
        _Pragma("clang loop unroll(full)")
        for (int b = 0; b < RB; b++) {
          if (FULL || b < nb) {
            uint4 xpk = sx[b * 32u + simd_lid];
            bfloat4 xv0 = as_type<bfloat4>(xpk.xy), xv1 = as_type<bfloat4>(xpk.zw);
            float x0 = float(xv0.x), x1 = float(xv0.y), x2 = float(xv0.z), x3 = float(xv0.w);
            float x4 = float(xv1.x), x5 = float(xv1.y), x6 = float(xv1.z), x7 = float(xv1.w);
            float smb = sm[b * 32u + simd_lid];
            _Pragma("clang loop unroll(full)")
            for (int r = 0; r < RPS; r++) {
              float acc = 0.0f;
              acc += x0 * float(w0[r].x); acc += x1 * float(w0[r].y); acc += x2 * float(w0[r].z); acc += x3 * float(w0[r].w);
              acc += x4 * float(w1[r].x); acc += x5 * float(w1[r].y); acc += x6 * float(w1[r].z); acc += x7 * float(w1[r].w);
              result[b][r] += sc[r] * acc + smb * bi[r];
            }
          }
        }
      }
    }
    if constexpr (!DBUF) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    ws += block_size;
    sc_p += 4;
    bi_p += 4;
    ap += block_size;
    gp += block_size;
  }
  if (!active) return;
  _Pragma("clang loop unroll(full)")
  for (int b = 0; b < RB; b++) {
    _Pragma("clang loop unroll(full)")
    for (int r = 0; r < RPS; r++) {
      float rr = simd_sum(result[b][r]);
      if (simd_lid == 0 && r <= last_row && (FULL || b < nb)) {
        yp[size_t(b) * N + r] = static_cast<bfloat16_t>(rr);
      }
    }
  }
"""

_oproj_gsx_k = None


def _get_oproj_gsx():
    global _oproj_gsx_k
    if _oproj_gsx_k is None:
        _oproj_gsx_k = mx.fast.metal_kernel(
            name="f3_oproj_gate_sx",
            input_names=["a", "g", "w", "scales", "biases"],
            output_names=["y"],
            header=GLUE_HEADER,
            source=_OPROJ_GSX_SRC,
            ensure_row_contiguous=True,
        )
    return _oproj_gsx_k


def _oproj_gsx_cfg(B):
    if B == 1:
        return 4, 2, 1, 0, 1
    if B == 2:
        return 4, 4, 2, 0, 1
    if B == 3:
        return 4, 4, 3, 0, 1
    if B == 4:
        return 4, 2, 4, 1, 1
    if B <= 8:
        return 4, 4, B, 1, 1
    if B % 4 == 0 and B % 8 != 0:
        return 4, 4, 4, 0, 1
    return 4, 4, 8, 1, 0


def oproj_rows(attn_out, gate, w, scales, biases):
    """K9. attn_out (B,3072) or (B,1,24,128) bf16, gate (B,3072) bf16 -> (B,2304) bf16;
    row b == o_proj(bf16(f32(attn_out[b]) * sigmoid(f32(gate[b])))) at M=1."""
    B = gate.shape[0]
    K = gate.shape[-1]
    if attn_out.size != B * K or attn_out.dtype != mx.bfloat16 or gate.dtype != mx.bfloat16:
        raise ValueError("attn_out and gate must be bf16 with B*K elements")
    a2 = attn_out.reshape(B, K)
    g2 = gate.reshape(B, K)
    N = w.shape[0]
    if (
        os.environ.get("LOOPLING_OPROJ_GSX", "1") not in ("0", "", "false", "False")
        and not os.environ.get("LOOPLING_QMV_CFG")
        and 1 <= B <= MAX_B
        and K % _qmv._K_ALIGN == 0
        and K not in (64, 128)
        and w.shape[1] * 4 == K
        and scales.shape == (N, K // 64)
        and biases.shape == (N, K // 64)
        and scales.dtype == mx.bfloat16
        and biases.dtype == mx.bfloat16
    ):
        rps, nsg, rb, packw, dbuf = _oproj_gsx_cfg(B)
        tiles = (N + rps * nsg - 1) // (rps * nsg)
        pr = (N % (rps * nsg)) != 0
        return _get_oproj_gsx()(
            inputs=[a2, g2, w, scales, biases],
            template=[
                ("B", B), ("K", K), ("N", N), ("PR", pr),
                ("RPS", rps), ("NSG", nsg), ("RB", rb), ("PACKW", packw), ("DBUF", dbuf),
            ],
            grid=(((B + rb - 1) // rb) * 32, nsg * tiles, 1),
            threadgroup=(32, nsg, 1),
            output_shapes=[(B, N)],
            output_dtypes=_OUT_DT_BF,
        )[0]
    return _launch_q([a2], [(w, scales, biases)], gate=g2).reshape(B, N)


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


def _gd_tm1_src(G, B, RB):
    wsel = "\n".join(
        ("%s (g == %d) wr = w%d;" % ("if" if g == 0 else "else if", g, g)) if g < G - 1 else ("else wr = w%d;" % g)
        for g in range(G)
    )
    if G == 1:
        wsel = "const device bfloat16_t* wr = w0;"
    else:
        wsel = "const device bfloat16_t* wr;\n    " + wsel
    full = (B % RB) == 0
    b_guard = "" if full else "if (b < nb) "
    if RB <= 4:
        k_loop = f"""    bfloat4 wv[4];
    wv[0] = as_type<bfloat4>(((const device uint2*)(wr + 0))[0]);
    wv[1] = as_type<bfloat4>(((const device uint2*)(wr + 1024))[0]);
    wv[2] = as_type<bfloat4>(((const device uint2*)(wr + 2048))[0]);
    wv[3] = as_type<bfloat4>(((const device uint2*)(wr + 3072))[0]);
    #pragma unroll
    for (int it = 0; it < 4; it++) {{
      const int off = it * 1024;
      const float4 wvf = float4(wv[it].x, wv[it].y, wv[it].z, wv[it].w);
      for (int b = 0; b < {RB}; b++) {{
        {b_guard}{{
          const device bfloat16_t* xp = xb0 + (size_t)b * xbs + off;
          bfloat4 xv = as_type<bfloat4>(((const device uint2*)xp)[0]);
          result[b] += wvf.x * static_cast<float>(xv.x);
          result[b] += wvf.y * static_cast<float>(xv.y);
          result[b] += wvf.z * static_cast<float>(xv.z);
          result[b] += wvf.w * static_cast<float>(xv.w);
        }}
      }}
    }}"""
    else:
        k_loop = f"""    for (int it = 0; it < 4; it++) {{
      const int off = it * 1024;
      bfloat4 wvt = as_type<bfloat4>(((const device uint2*)(wr + off))[0]);
      const float4 wvf = float4(wvt.x, wvt.y, wvt.z, wvt.w);
      for (int b = 0; b < {RB}; b++) {{
        {b_guard}{{
          const device bfloat16_t* xp = xb0 + (size_t)b * xbs + off;
          bfloat4 xv = as_type<bfloat4>(((const device uint2*)xp)[0]);
          result[b] += wvf.x * static_cast<float>(xv.x);
          result[b] += wvf.y * static_cast<float>(xv.y);
          result[b] += wvf.z * static_cast<float>(xv.z);
          result[b] += wvf.w * static_cast<float>(xv.w);
        }}
      }}
    }}"""
    b0_decl = (
        "    const int b0 = 0;\n    const int nb = %d; (void)nb;" % RB
        if B == RB
        else f"    const int b0 = (int)threadgroup_position_in_grid.z * {RB};\n    const int nb = {'%d' % RB if full else f'min({RB}, {B} - b0)'}; (void)nb;"
    )
    return f"""
    const int t = (int)threadgroup_position_in_grid.x;
    const int g = t / 64;
    const int row = t % 64;
    const int sgN = (int)simdgroup_index_in_threadgroup;
    const int thrN = (int)thread_index_in_simdgroup;
    const int bn0 = (sgN * 32 + thrN) * 4;
{b0_decl}
    {wsel}
    wr += (size_t)row * 4608 + bn0;
    const size_t xbs = (size_t){G} * 4608;
    const device bfloat16_t* xb0 = x + (size_t)b0 * xbs + (size_t)g * 4608 + bn0;
    float result[{RB}];
    for (int b = 0; b < {RB}; b++) result[b] = 0.0f;
{k_loop}
    if (bn0 + 4096 < 4608) {{
      const int off = 4096;
      const device bfloat16_t* wp = wr + off;
      bfloat4 wvt = as_type<bfloat4>(((const device uint2*)wp)[0]);
      const float4 wvf = float4(wvt.x, wvt.y, wvt.z, wvt.w);
      for (int b = 0; b < {RB}; b++) {{
        {b_guard}{{
          const device bfloat16_t* xp = xb0 + (size_t)b * xbs + off;
          bfloat4 xv = as_type<bfloat4>(((const device uint2*)xp)[0]);
          result[b] += wvf.x * static_cast<float>(xv.x);
          result[b] += wvf.y * static_cast<float>(xv.y);
          result[b] += wvf.z * static_cast<float>(xv.z);
          result[b] += wvf.w * static_cast<float>(xv.w);
        }}
      }}
    }}
    for (int b = 0; b < {RB}; b++) {{
      for (ushort sn = 16; sn >= 1; sn >>= 1) result[b] += simd_shuffle_down(result[b], sn);
    }}
    threadgroup float tgp[{RB} * 8];
    if (thrN == 0) {{
      for (int b = 0; b < {RB}; b++) tgp[b * 8 + sgN] = result[b];
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgN == 0 && thrN == 0) {{
      for (int b = 0; b < {RB}; b++) {{
        {b_guard}{{
          float s = result[b];
          for (int k = 1; k < 8; k++) s += tgp[b * 8 + k];
          o[((size_t)(b0 + b) * {G} + g) * 64 + row] = static_cast<bfloat16_t>(s);
        }}
      }}
    }}
"""


def gate_down_rows(x, ws):
    """K2. x (B,G,K) bf16 (row b, gate g = the 4608-vector of gate g), ws = G weights (64,K) bf16 as stored
    -> (B,G,64) bf16, == (x[b,g] @ ws[g].T) at M=1 for every (b,g); one launch."""
    B, G, K = x.shape
    if x.dtype != mx.bfloat16 or len(ws) != G or not 1 <= B <= MAX_B:
        raise ValueError(f"x must be (B,G,K) bf16 with one weight per gate, B 1..{MAX_B}")
    N = ws[0].shape[0]
    for w in ws:
        if w.shape != (N, K) or w.dtype != mx.bfloat16:
            raise ValueError("weights must all be (N,K) bf16")
    if _GD_TM1 and N == 64 and K == 4608 and B <= MAX_B:
        ov_rb = os.environ.get("LOOPLING_GD_RB")
        rb = min(int(ov_rb), B) if ov_rb else min(B, 8)
        key = ("tm1", B, rb, G, N, K)
        ent = _k2.get(key)
        if ent is None:
            kern = mx.fast.metal_kernel(
                name="f3_gate_down_tm1_b%d_rb%d_g%d" % (B, rb, G),
                input_names=["x"] + ["w%d" % g for g in range(G)],
                output_names=["o"],
                source=_gd_tm1_src(G, B, rb),
                ensure_row_contiguous=True,
            )
            ent = _k2[key] = (kern, (G * 64 * 32, 8, (B + rb - 1) // rb), (32, 8, 1), [(B, G, N)])
        kern, grid, tg, out_shp = ent
        return kern(
            inputs=[x] + list(ws),
            grid=grid,
            threadgroup=tg,
            output_shapes=out_shp,
            output_dtypes=_OUT_DT_BF,
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


_MSG_BF16 = os.environ.get("LOOPLING_MSG_BF16", "1") not in ("0", "", "false", "False")
_BF16_W_CACHE: dict[int, tuple[mx.array, mx.array | None]] = {}


_BF16_MASK = mx.array(0xFFFF, mx.uint32)


def _maybe_bf16_w(w: mx.array) -> mx.array | None:
    """Return bit-exact BF16 copy of an FP32 weight (identity-guarded) if lossless and LOOPLING_MSG_BF16=1."""
    if w.dtype == mx.bfloat16:
        return w
    if not _MSG_BF16 or w.dtype != mx.float32:
        return None
    wid = id(w)
    entry = _BF16_W_CACHE.get(wid)
    if entry is not None and entry[0] is w:
        return entry[1]
    lossless = bool(mx.all((w.view(mx.uint32) & _BF16_MASK) == 0).item())
    val = w.astype(mx.bfloat16) if lossless else None
    if val is not None:
        mx.eval(val)
    if len(_BF16_W_CACHE) >= 512:
        _BF16_W_CACHE.clear()
    _BF16_W_CACHE[wid] = (w, val)
    return val


# =============================================================================================== K3 (gate mix)
# forget/message_gate_proj (D=2304, 64) f32 at M=1: MLX gemv bm1 bn1 sm8 sn4 tm4 tn4 -> each quad of 4 lanes owns 1 row
# (K=64 split 4 lanes x 4 uint4/float4 loads), 8 rows per simdgroup, 4 simdgroups (32 output rows) per threadgroup;
# forget + message weights share one K-loop and reduce with shuffle_down(2), (1); lane (lane&3)==0 writes the epilogue.
_H3 = r"""
inline float proj_inv_from_part(const device float* p, float eps, uint lane) {
  GL_NOFMA
  float pv = (lane < 18u) ? p[lane] : 0.0f;
  float tot = simd_sum(pv);
  return metal::precise::rsqrt(tot / 2304.0f + eps);
}

template <typename WT>
inline float4 load_w4(const device WT* p) {
  if constexpr (sizeof(WT) == 2) {
    return float4(as_type<bfloat4>(((const device uint2*)p)[0]));
  } else {
    return ((const device float4*)p)[0];
  }
}

template <int B, int RB, int G, int D, int NSG, typename WT = float>
METAL_FUNC void gate_mix_one(const device WT* wf, const device WT* wm, const device bfloat16_t* down, const int g,
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
  const device WT* wfr = wf + (size_t)row * 64 + qn;
  const device WT* wmr = wm + (size_t)row * 64 + qn;
  if constexpr (B <= 4) {
    float4 ffa[4], mma[4];
    ffa[0] = load_w4(wfr + 0);
    ffa[1] = load_w4(wfr + 16);
    ffa[2] = load_w4(wfr + 32);
    ffa[3] = load_w4(wfr + 48);
    mma[0] = load_w4(wmr + 0);
    mma[1] = load_w4(wmr + 16);
    mma[2] = load_w4(wmr + 32);
    mma[3] = load_w4(wmr + 48);
    #pragma unroll
    for (int i = 0; i < 4; i++) {
      int off = i * 16;
      float4 ff = ffa[i];
      float4 mm = mma[i];
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
  } else {
    for (int p = 0; p < 2; p++) {
      int o0 = p * 32, o1 = p * 32 + 16;
      float4 f0 = load_w4(wfr + o0);
      float4 f1 = load_w4(wfr + o1);
      float4 m0 = load_w4(wmr + o0);
      float4 m1 = load_w4(wmr + o1);
      #pragma unroll
      for (int b = 0; b < RB; b++) {
        if (b < nb) {
          bfloat4 xv0 = as_type<bfloat4>(((const threadgroup uint2*)(dn_s + b * 64 + qn + o0))[0]);
          float v0 = float(xv0.x), v1 = float(xv0.y), v2 = float(xv0.z), v3 = float(xv0.w);
          fr[b] += f0.x * v0; fr[b] += f0.y * v1; fr[b] += f0.z * v2; fr[b] += f0.w * v3;
          mr[b] += m0.x * v0; mr[b] += m0.y * v1; mr[b] += m0.z * v2; mr[b] += m0.w * v3;
          bfloat4 xv1 = as_type<bfloat4>(((const threadgroup uint2*)(dn_s + b * 64 + qn + o1))[0]);
          float u0 = float(xv1.x), u1 = float(xv1.y), u2 = float(xv1.z), u3 = float(xv1.w);
          fr[b] += f1.x * u0; fr[b] += f1.y * u1; fr[b] += f1.z * u2; fr[b] += f1.w * u3;
          mr[b] += m1.x * u0; mr[b] += m1.y * u1; mr[b] += m1.z * u2; mr[b] += m1.w * u3;
        }
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

// K3 + exact 36-half-SG rms_single_row partial sum-of-squares (36 TGs/gate = 108 TGs for G=3, 64 rows/TG, NSG=8, 1 pass).
template <int B, int RB, int G, int D, int NSG, typename WT = float>
METAL_FUNC void gate_mix_part_one(const device WT* wf, const device WT* wm, const device bfloat16_t* down, const int g,
                                  const device float* s, const device float* msg, const int slot,
                                  device float* out, device float* part,
                                  const uint vs, const uint b0, const uint sg, const uint lane,
                                  threadgroup bfloat16_t* dn_s, threadgroup float* mix_s) {
  static_assert(D == 2304 && NSG == 8, "gate_mix_part_one expects D=2304, NSG=8");
  const int nb = min(RB, B - (int)b0);
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
  const uint row_base = vs * 64u;
  const int local_row = int(sg * 8u + (lane >> 2));
  const int row = int(row_base) + local_row;
  float fr[RB], mr[RB];
  for (int b = 0; b < RB; b++) {
    fr[b] = 0.0f;
    mr[b] = 0.0f;
  }
  const device WT* wfr = wf + (size_t)row * 64 + qn;
  const device WT* wmr = wm + (size_t)row * 64 + qn;
  if constexpr (B <= 4) {
    float4 ffa[4], mma[4];
    ffa[0] = load_w4(wfr + 0);
    ffa[1] = load_w4(wfr + 16);
    ffa[2] = load_w4(wfr + 32);
    ffa[3] = load_w4(wfr + 48);
    mma[0] = load_w4(wmr + 0);
    mma[1] = load_w4(wmr + 16);
    mma[2] = load_w4(wmr + 32);
    mma[3] = load_w4(wmr + 48);
    #pragma unroll
    for (int i = 0; i < 4; i++) {
      int off = i * 16;
      float4 ff = ffa[i];
      float4 mm = mma[i];
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
  } else {
    for (int p = 0; p < 2; p++) {
      int o0 = p * 32, o1 = p * 32 + 16;
      float4 f0 = load_w4(wfr + o0);
      float4 f1 = load_w4(wfr + o1);
      float4 m0 = load_w4(wmr + o0);
      float4 m1 = load_w4(wmr + o1);
      #pragma unroll
      for (int b = 0; b < RB; b++) {
        if (b < nb) {
          bfloat4 xv0 = as_type<bfloat4>(((const threadgroup uint2*)(dn_s + b * 64 + qn + o0))[0]);
          float v0 = float(xv0.x), v1 = float(xv0.y), v2 = float(xv0.z), v3 = float(xv0.w);
          fr[b] += f0.x * v0; fr[b] += f0.y * v1; fr[b] += f0.z * v2; fr[b] += f0.w * v3;
          mr[b] += m0.x * v0; mr[b] += m0.y * v1; mr[b] += m0.z * v2; mr[b] += m0.w * v3;
          bfloat4 xv1 = as_type<bfloat4>(((const threadgroup uint2*)(dn_s + b * 64 + qn + o1))[0]);
          float u0 = float(xv1.x), u1 = float(xv1.y), u2 = float(xv1.z), u3 = float(xv1.w);
          fr[b] += f1.x * u0; fr[b] += f1.y * u1; fr[b] += f1.z * u2; fr[b] += f1.w * u3;
          mr[b] += m1.x * u0; mr[b] += m1.y * u1; mr[b] += m1.z * u2; mr[b] += m1.w * u3;
        }
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
        float mv = glue_msg_combine<0>(fr[b], mr[b], s[gb * D + row], msg[(gb * 4 + slot) * D + row]);
        out[(gb * G + g) * D + row] = mv;
        mix_s[b * 64 + local_row] = mv;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int b = (int)sg; b < nb; b += NSG) {
    GL_NOFMA
    float acc = 0.0f;
    if (lane < 16u) {
      float4 v = ((const threadgroup float4*)(mix_s + b * 64 + lane * 4))[0];
      acc = fma(v.x, v.x, acc);
      acc = fma(v.y, v.y, acc);
      acc = fma(v.z, v.z, acc);
      acc = fma(v.w, v.w, acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) {
      part[((size_t)(b0 + b) * G + g) * 36u + vs] = acc;
    }
  }
}
"""

_k3 = {}
_k3p = {}
_GMIX_RB_ENV = os.environ.get("LOOPLING_GMIX_RB")


def _gmix_rb(B):
    if _GMIX_RB_ENV is not None:
        return min(B, max(1, int(_GMIX_RB_ENV)))
    return min(B, 2)


_MIX_W_CACHE: dict = {}


def _prepare_mix_inputs(ss, msg, wf, wm, D):
    if msg.shape[1:] != (4, D) or msg.dtype != mx.float32:
        raise ValueError("msg must be (B,4,D) f32")
    want_s = (msg.shape[0], D)
    G = len(ss)
    if G == 1:
        s0 = ss[0]
        if s0.shape != want_s or s0.dtype != mx.float32:
            raise ValueError("each s must be (B,D) f32")
        uniq, smap = [s0], (0,)
        w0, w1 = wf[0], wm[0]
        wkey = (id(w0), id(w1))
        ent = _MIX_W_CACHE.get(wkey)
        if ent is not None and ent[0][0] is w0 and ent[0][1] is w1:
            w_ins, use_bf16, wt = ent[1]
            return uniq, smap, w_ins, use_bf16, wt
        w_refs = (w0, w1)
    elif G == 3:
        s0, s1, s2 = ss[0], ss[1], ss[2]
        if s0 is s1 and s1 is s2:
            if s0.shape != want_s or s0.dtype != mx.float32:
                raise ValueError("each s must be (B,D) f32")
            uniq, smap = [s0], (0, 0, 0)
        else:
            if (s0.shape != want_s or s0.dtype != mx.float32
                or s1.shape != want_s or s1.dtype != mx.float32
                or s2.shape != want_s or s2.dtype != mx.float32):
                raise ValueError("each s must be (B,D) f32")
            uniq, smap = [s0], [0]
            if s1 is s0:
                smap.append(0)
            else:
                smap.append(len(uniq))
                uniq.append(s1)
            if s2 is s0:
                smap.append(0)
            elif len(uniq) > 1 and s2 is uniq[1]:
                smap.append(1)
            else:
                smap.append(len(uniq))
                uniq.append(s2)
            smap = tuple(smap)
        w0, w1, w2 = wf[0], wf[1], wf[2]
        w3, w4, w5 = wm[0], wm[1], wm[2]
        wkey = (id(w0), id(w1), id(w2), id(w3), id(w4), id(w5))
        ent = _MIX_W_CACHE.get(wkey)
        if ent is not None:
            er = ent[0]
            if er[0] is w0 and er[1] is w1 and er[2] is w2 and er[3] is w3 and er[4] is w4 and er[5] is w5:
                w_ins, use_bf16, wt = ent[1]
                return uniq, smap, w_ins, use_bf16, wt
        w_refs = (w0, w1, w2, w3, w4, w5)
    else:
        uniq, smap_l = [], []
        for s in ss:
            if s.shape != want_s or s.dtype != mx.float32:
                raise ValueError("each s must be (B,D) f32")
            for j, u in enumerate(uniq):
                if u is s:
                    smap_l.append(j)
                    break
            else:
                smap_l.append(len(uniq))
                uniq.append(s)
        smap = tuple(smap_l)
        w_refs = tuple(wf) + tuple(wm)
        wkey = tuple(id(w) for w in w_refs)
        ent = _MIX_W_CACHE.get(wkey)
        if ent is not None and len(ent[0]) == len(w_refs) and all(x is y for x, y in zip(ent[0], w_refs)):
            w_ins, use_bf16, wt = ent[1]
            return uniq, smap, w_ins, use_bf16, wt
    for w in w_refs:
        if w.shape != (D, 64) or w.dtype not in (mx.float32, mx.bfloat16):
            raise ValueError("forget/message weights must be (D,64) f32 or bf16")
    wf_b = [_maybe_bf16_w(w) for w in wf]
    wm_b = [_maybe_bf16_w(w) for w in wm]
    use_bf16 = all(x is not None for x in wf_b) and all(x is not None for x in wm_b)
    if not use_bf16 and any(w.dtype == mx.bfloat16 for w in w_refs):
        wf = [w.astype(mx.float32) if w.dtype == mx.bfloat16 else w for w in wf]
        wm = [w.astype(mx.float32) if w.dtype == mx.bfloat16 else w for w in wm]
    w_ins = (wf_b + wm_b) if use_bf16 else (list(wf) + list(wm))
    wt = "bfloat16_t" if use_bf16 else "float"
    if len(_MIX_W_CACHE) >= 256:
        _MIX_W_CACHE.clear()
    _MIX_W_CACHE[wkey] = (w_refs, (w_ins, use_bf16, wt))
    return uniq, smap, w_ins, use_bf16, wt


def gate_mix_part_rows(down, ss, msg, slots, wf, wm):
    """K3 + exact 18-virtual-SG RMS partials -> (mixed (B,G,2304) f32, part (B,G,18) f32).
    Eliminates K3b launch when consumer (K4 qkv_rows_from_mixed / K11+K12) normalizes on load."""
    B, G, R = down.shape
    D = wf[0].shape[0]
    if D != 2304 or R != 64 or down.dtype != mx.bfloat16 or not 1 <= B <= MAX_B or len(ss) != G or len(wf) != G or len(wm) != G or msg.shape[0] != B:
        raise ValueError(f"gate_mix_part_rows requires D=2304, R=64, down bf16, B in 1..{MAX_B}")
    uniq, smap, w_ins, use_bf16, wt = _prepare_mix_inputs(ss, msg, wf, wm, D)
    nsg = 8
    nt = 36  # 2304 / 64 = 36 half-SG tiles (108 TGs for G=3 -> all 80 M3 Ultra cores busy, 1 pass!)
    rb = _gmix_rb(B)
    key = (B, rb, G, smap, slots, use_bf16)
    ent = _k3p.get(key)
    if ent is None:
        lines = [
            f"threadgroup bfloat16_t dn_s[{rb} * 64];",
            f"threadgroup float mix_s[{rb} * 64];",
            "const uint t = threadgroup_position_in_grid.y;",
            f"const uint b0 = threadgroup_position_in_grid.x * {rb};",
            "const uint sg = simdgroup_index_in_threadgroup;",
            "const uint lane = thread_index_in_simdgroup;",
            f"const uint vs = t % {nt};",
        ]
        for g in range(G):
            if g < G - 1:
                head = ("if" if g == 0 else "else if") + f" (t < {(g + 1) * nt})"
            else:
                head = "else" if g else ""
            lines.append(
                "%s { gate_mix_part_one<%d, %d, %d, %d, %d, %s>(wf%d, wm%d, down, %d, s%d, msg, %d, out, part, vs, b0, sg, lane, dn_s, mix_s); }"
                % (head, B, rb, G, D, nsg, wt, g, g, g, smap[g], slots[g])
            )
        kern = mx.fast.metal_kernel(
            name="f3_gate_mix_part36_b%d_rb%d_g%d_s%s_m%s%s"
            % (B, rb, G, "".join(map(str, smap)), "".join(map(str, slots)), "_bf16" if use_bf16 else ""),
            input_names=["down", "msg"] + ["s%d" % j for j in range(len(uniq))]
            + ["wf%d" % g for g in range(G)] + ["wm%d" % g for g in range(G)],
            output_names=["out", "part"],
            header=GLUE_HEADER + _H3,
            source="\n".join(lines),
            ensure_row_contiguous=True,
        )
        tg_t = nsg * 32
        ent = _k3p[key] = (kern, (((B + rb - 1) // rb) * tg_t, G * nt, 1), (tg_t, 1, 1), [(B, G, D), (B, G, 36)])
    kern, grid, tg, out_shp = ent
    return kern(
        inputs=[down, msg] + uniq + w_ins,
        grid=grid,
        threadgroup=tg,
        output_shapes=out_shp,
        output_dtypes=_OUT_DT_F32_2,
    )


def gate_mix_rows(down, ss, msg, slots, wf, wm):
    """K3. down (B,G,64) bf16 (K2 output); ss = list of G arrays (B,D) f32 (the hidden state the gate mixes into; the same
    array may repeat); msg (B,4,D) f32; slots = G message indices (q,k,v = 0,1,2; mlp = 3); wf/wm = G forget/message
    weights (D,64) f32.  Returns (B,G,D) f32:  (1 - tanh(wf@f32(down))) * s + tanh(wm@f32(down)) * msg[slot], exactly the
    MessageGate epilogue (separate mul/add, no fma)."""
    B, G, R = down.shape
    D = wf[0].shape[0]
    if R != 64 or down.dtype != mx.bfloat16 or not 1 <= B <= MAX_B or len(ss) != G or len(wf) != G or len(wm) != G or msg.shape[0] != B:
        raise ValueError("down must be (B,G,64) bf16; one s/wf/wm per gate")
    if D % 32 or (D != 2304 and _gemv._cfg(D, 64) != (1, 1, 8, 4, 4, 4)):
        raise ValueError("gate_mix_rows is specialised for D % 32 == 0 and MLX's bm1 bn1 sm8 sn4 tm4 tn4 gemv config")
    uniq, smap, w_ins, use_bf16, wt = _prepare_mix_inputs(ss, msg, wf, wm, D)
    nsg = 2 if (G == 1 and D % 16 == 0) else 4
    rows_tg = nsg * 8
    nt = D // rows_tg
    rb = _gmix_rb(B)
    key = (B, rb, G, D, nsg, smap, slots, use_bf16)
    ent = _k3.get(key)
    if ent is None:
        lines = [
            f"threadgroup bfloat16_t dn_s[{rb} * 64];",
            "const uint t = threadgroup_position_in_grid.y;",
            f"const uint b0 = threadgroup_position_in_grid.x * {rb};",
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
                "%s { gate_mix_one<%d, %d, %d, %d, %d, %s>(wf%d, wm%d, down, %d, s%d, msg, %d, out, row0, b0, sg, lane, dn_s); }"
                % (head, B, rb, G, D, nsg, wt, g, g, g, smap[g], slots[g])
            )
        kern = mx.fast.metal_kernel(
            name="f3_gate_mix_b%d_rb%d_g%d_n%d_s%s_m%s%s"
            % (B, rb, G, nsg, "".join(map(str, smap)), "".join(map(str, slots)), "_bf16" if use_bf16 else ""),
            input_names=["down", "msg"] + ["s%d" % j for j in range(len(uniq))]
            + ["wf%d" % g for g in range(G)] + ["wm%d" % g for g in range(G)],
            output_names=["out"],
            header=GLUE_HEADER + _H3,
            source="\n".join(lines),
            ensure_row_contiguous=True,
        )
        tg_t = nsg * 32
        ent = _k3[key] = (kern, (((B + rb - 1) // rb) * tg_t, G * nt, 1), (tg_t, 1, 1), [(B, G, D)], [mx.float32])
    kern, grid, tg, out_shp, out_dt = ent
    return kern(
        inputs=[down, msg] + uniq + w_ins,
        grid=grid,
        threadgroup=tg,
        output_shapes=out_shp,
        output_dtypes=out_dt,
    )[0]
