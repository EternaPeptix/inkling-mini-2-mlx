# Copyright © 2026 Apple Inc.

import os

import mlx.core as mx

from mlx_lm.kernels.common import row_threads, scalar

__all__ = [
    "depth_attention",
    "depth_attention_batch",
    "relative_attention",
]


# Attention of learned queries over the depth sources of each token, in one
# kernel.
_DEPTH_ATTENTION_SOURCE = """
    constexpr int N = K + 1;
    constexpr int MAX_IT = (D + TG - 1) / TG;
    uint row = threadgroup_position_in_grid.x;
    uint rows = threadgroups_per_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;

    float acc[N * Q];
    for (int i = 0; i < N * Q; i++) {
        acc[i] = 0.0f;
    }
    float xc[MAX_IT][N];
    int it_idx = 0;
    for (int d = tid; d < D; d += TG, it_idx++) {
        for (int i = 0; i < K; i++) {
            xc[it_idx][i] = blocks[(size_t(i) * rows + row) * D + d];
        }
        xc[it_idx][K] = partial[size_t(row) * D + d];
        for (int j = 0; j < Q; j++) {
            float qv = queries[j * D + d];
            for (int i = 0; i < N; i++) {
                acc[i * Q + j] += xc[it_idx][i] * qv;
            }
        }
    }

    threadgroup float buf[N * Q][TG / 32];
    for (int i = 0; i < N * Q; i++) {
        float v = simd_sum(acc[i]);
        if (lane == 0) {
            buf[i][sg] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float w[N * Q];
    for (int i = 0; i < N * Q; i++) {
        float t = 0.0f;
        for (int s = 0; s < TG / 32; s++) {
            t += buf[i][s];
        }
        w[i] = t * scale;
    }
    for (int j = 0; j < Q; j++) {
        float m = -INFINITY;
        for (int i = 0; i < N; i++) {
            m = max(m, w[i * Q + j]);
        }
        float z = 0.0f;
        for (int i = 0; i < N; i++) {
            w[i * Q + j] = metal::exp(w[i * Q + j] - m);
            z += w[i * Q + j];
        }
        for (int i = 0; i < N; i++) {
            w[i * Q + j] /= z;
        }
    }

    it_idx = 0;
    for (int d = tid; d < D; d += TG, it_idx++) {
        for (int j = 0; j < Q; j++) {
            float o = 0.0f;
            for (int i = 0; i < N; i++) {
                o += w[i * Q + j] * xc[it_idx][i];
            }
            out[(size_t(row) * Q + j) * D + d] = o;
        }
    }
"""

_depth_attention_kernel = mx.fast.metal_kernel(
    name="depth_attention",
    input_names=["blocks", "partial", "queries", "scale"],
    output_names=["out"],
    source=_DEPTH_ATTENTION_SOURCE,
)


def depth_attention(
    blocks: mx.array, partial: mx.array, queries: mx.array, *, scale: float
) -> mx.array:
    """blocks: (K, ..., D) and partial: (..., D), float32, also the values.
    queries: float32 (Q, D). Returns (..., Q, D)."""
    D = partial.shape[-1]
    Q = queries.shape[0]
    TG = row_threads(D)
    return _depth_attention_kernel(
        inputs=[blocks, partial, queries, scalar(scale)],
        template=[("K", blocks.shape[0]), ("D", D), ("Q", Q), ("TG", TG)],
        grid=(partial.size // D * TG, 1, 1),
        threadgroup=(TG, 1, 1),
        output_shapes=[partial.shape[:-1] + (Q, D)],
        output_dtypes=[mx.float32],
    )[0]


_DEPTH_ATTENTION_BATCH_SOURCE = (
    _DEPTH_ATTENTION_SOURCE
    .replace(
        "uint sg = simdgroup_index_in_threadgroup;\n",
        "uint sg = simdgroup_index_in_threadgroup;\n    uint ly = threadgroup_position_in_grid.y;\n",
    )
    .replace("queries[j * D + d]", "queries[((size_t)ly * Q + j) * D + d]")
    .replace(
        "out[(size_t(row) * Q + j) * D + d] = o;",
        "out[(((size_t)ly * rows + row) * Q + j) * D + d] = o;",
    )
)
assert _DEPTH_ATTENTION_BATCH_SOURCE != _DEPTH_ATTENTION_SOURCE

_depth_attention_batch_kernel = mx.fast.metal_kernel(
    name="depth_attention_batch",
    input_names=["blocks", "partial", "queries", "scale"],
    output_names=["out"],
    source=_DEPTH_ATTENTION_BATCH_SOURCE,
)

_DEPTH_ATTENTION_BATCH_NL_SOURCE = """
    constexpr int N = K + 1;
    constexpr int MAX_IT = (D + TG - 1) / TG;
    uint row = threadgroup_position_in_grid.x;
    uint rows = threadgroups_per_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;

    float xc[MAX_IT][N];
    int it_idx = 0;
    for (int d = tid; d < D; d += TG, it_idx++) {
        for (int i = 0; i < K; i++) xc[it_idx][i] = blocks[(size_t(i) * rows + row) * D + d];
        xc[it_idx][K] = partial[size_t(row) * D + d];
    }

    threadgroup float buf[N * Q][TG / 32];
    for (int ly = 0; ly < NL; ly++) {
        float acc[N * Q];
        for (int i = 0; i < N * Q; i++) acc[i] = 0.0f;
        it_idx = 0;
        for (int d = tid; d < D; d += TG, it_idx++) {
            for (int j = 0; j < Q; j++) {
                float qv = queries[((size_t)ly * Q + j) * D + d];
                for (int i = 0; i < N; i++) acc[i * Q + j] += xc[it_idx][i] * qv;
            }
        }
        for (int i = 0; i < N * Q; i++) {
            float v = simd_sum(acc[i]);
            if (lane == 0) buf[i][sg] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float w[N * Q];
        for (int i = 0; i < N * Q; i++) {
            float t = 0.0f;
            for (int s = 0; s < TG / 32; s++) t += buf[i][s];
            w[i] = t * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int j = 0; j < Q; j++) {
            float m = -INFINITY;
            for (int i = 0; i < N; i++) m = max(m, w[i * Q + j]);
            float z = 0.0f;
            for (int i = 0; i < N; i++) { w[i * Q + j] = metal::exp(w[i * Q + j] - m); z += w[i * Q + j]; }
            for (int i = 0; i < N; i++) w[i * Q + j] /= z;
        }
        it_idx = 0;
        for (int d = tid; d < D; d += TG, it_idx++) {
            for (int j = 0; j < Q; j++) {
                float o = 0.0f;
                for (int i = 0; i < N; i++) o += w[i * Q + j] * xc[it_idx][i];
                out[(((size_t)ly * rows + row) * Q + j) * D + d] = o;
            }
        }
    }
"""

_depth_attention_batch_nl_kernel = mx.fast.metal_kernel(
    name="depth_attention_batch_nl",
    input_names=["blocks", "partial", "queries", "scale"],
    output_names=["out"],
    source=_DEPTH_ATTENTION_BATCH_NL_SOURCE,
)


def depth_attention_batch(
    blocks: mx.array, partial: mx.array, queries: mx.array, *, scale: float
) -> mx.array:
    """Batched 2D-grid depth_attention over queries (L, Q, D) f32 -> (L, ..., Q, D) f32 (exact per l)."""
    D = partial.shape[-1]
    L, Q = queries.shape[0], queries.shape[1]
    TG = row_threads(D)
    rows = partial.size // D
    if rows >= 64:
        return _depth_attention_batch_nl_kernel(
            inputs=[blocks, partial, queries, scalar(scale)],
            template=[("K", blocks.shape[0]), ("D", D), ("Q", Q), ("TG", TG), ("NL", L)],
            grid=(rows * TG, 1, 1),
            threadgroup=(TG, 1, 1),
            output_shapes=[(L,) + partial.shape[:-1] + (Q, D)],
            output_dtypes=[mx.float32],
        )[0]
    return _depth_attention_batch_kernel(
        inputs=[blocks, partial, queries, scalar(scale)],
        template=[("K", blocks.shape[0]), ("D", D), ("Q", Q), ("TG", TG)],
        grid=(rows * TG, L, 1),
        threadgroup=(TG, 1, 1),
        output_shapes=[(L,) + partial.shape[:-1] + (Q, D)],
        output_dtypes=[mx.float32],
    )[0]


# Causal flash attention with a relative position bias and a sink logit per
# query.
# Queries per threadgroup, 8 for each of its 4 simdgroups.
_BQ = 32

_TG = 128

# The online softmax of each row starts from its sink logit, so the zero-value
# sink needs no key.
_RELATIVE_ATTENTION_SOURCE = """
    constexpr float LOG2E = 1.4426950408889634f;
    constexpr int BK = 16;
    constexpr int LDK = BK + 8;
    constexpr int LDV = D + 8;
    const int L = sizes[0];
    const int S = sizes[1];
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const int batch = threadgroup_position_in_grid.z;
    const int head = threadgroup_position_in_grid.y;
    const int kv_head = head / (HQ / HKV);
    const int q0 = threadgroup_position_in_grid.x * BQ;
    const int start = key_start[batch];
    const float qscale = scale * LOG2E;

    // The lane holds row fm and columns fn, fn + 1 of each 8x8 fragment.
    const short fm = ((lane / 4) & 4) + (lane / 2) % 4;
    const short fn = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
    const int row = q0 + int(sg) * 8 + fm;
    const bool row_ok = row < L;
    const int diag = S - L + row;

    // q and k are clipped and v has unit RMS, so half holds them.
    threadgroup half Ks[D * LDK];
    threadgroup half Vs[BK * LDV];

    simdgroup_half8x8 q_frag[D / 8];
    for (int db = 0; db < D / 8; ++db) {
        half2 qv = half2(0.0h);
        if (row_ok) {
            const device T* qp = q + batch * q_strides[0] + head * q_strides[1]
                + row * q_strides[2] + db * 8 + fn;
            qv = half2(float(qp[0]), float(qp[1]));
        }
        q_frag[db].thread_elements()[0] = qv.x;
        q_frag[db].thread_elements()[1] = qv.y;
    }
    simdgroup_float8x8 acc[D / 8];
    for (int db = 0; db < D / 8; ++db) {
        acc[db] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    float row_max = row_ok ? float(sinks[(batch * L + row) * HQ + head]) * LOG2E : 0.0f;
    float row_sum = 1.0f;
    const int bias_row = batch * L + (row_ok ? row : 0);
    const device T* bias = rel + (size_t(bias_row) * HQ + head) * R;

    const int first_diag = S - L + q0;
    const int last_diag = S - L + min(q0 + BQ, L) - 1;
    int first_key = WINDOW > 0 ? max(0, first_diag - WINDOW + 1) : 0;
    first_key = max(first_key, start);
    const device T* k_head = k + batch * k_strides[0] + kv_head * k_strides[1];
    const device T* v_head = v + batch * v_strides[0] + kv_head * v_strides[1];

    for (int k0 = first_key / BK * BK; k0 <= last_diag; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int c = tid; c < BK * D / 8; c += TG) {
            const int r = c / (D / 8);
            const int col = (c % (D / 8)) * 8;
            const int key = k0 + r;
            vec<T, 4> kv[2] = {vec<T, 4>(0), vec<T, 4>(0)};
            vec<T, 4> vv[2] = {vec<T, 4>(0), vec<T, 4>(0)};
            if (key < S) {
                auto kp = (const device vec<T, 4>*)(k_head + key * k_strides[2] + col);
                auto vp = (const device vec<T, 4>*)(v_head + key * v_strides[2] + col);
                kv[0] = kp[0];
                kv[1] = kp[1];
                vv[0] = vp[0];
                vv[1] = vp[1];
            }
            for (int j = 0; j < 8; ++j) {
                Ks[(col + j) * LDK + r] = half(float(kv[j / 4][j % 4]));
                Vs[r * LDV + col + j] = half(float(vv[j / 4][j % 4]));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (q0 + int(sg) * 8 >= L) {
            continue;
        }

        simdgroup_float8x8 scores[BK / 8];
        for (int kb = 0; kb < BK / 8; ++kb) {
            scores[kb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int db = 0; db < D / 8; ++db) {
                simdgroup_half8x8 k_frag;
                simdgroup_load(k_frag, Ks + db * 8 * LDK + kb * 8, LDK);
                simdgroup_multiply_accumulate(scores[kb], q_frag[db], k_frag, scores[kb]);
            }
        }

        float tile_max = -INFINITY;
        for (int kb = 0; kb < BK / 8; ++kb) {
            for (int e = 0; e < 2; ++e) {
                const int key = k0 + kb * 8 + fn + e;
                const int d = diag - key;
                const bool visible = row_ok && d >= 0 && key >= start
                    && (WINDOW == 0 || d < WINDOW);
                float s = -INFINITY;
                if (visible) {
                    s = scores[kb].thread_elements()[e] * qscale;
                    if (d < R) {
                        s += float(bias[d]) * LOG2E;
                    }
                }
                scores[kb].thread_elements()[e] = s;
                tile_max = max(tile_max, s);
            }
        }
        // The 4 lanes of a row are lane ^ 1 and lane ^ 8 apart.
        tile_max = max(tile_max, simd_shuffle_xor(tile_max, 1));
        tile_max = max(tile_max, simd_shuffle_xor(tile_max, 8));
        const float new_max = max(row_max, tile_max);
        const float factor = fast::exp2(row_max - new_max);
        float tile_sum = 0.0f;
        simdgroup_half8x8 probs[BK / 8];
        for (int kb = 0; kb < BK / 8; ++kb) {
            for (int e = 0; e < 2; ++e) {
                const float p = fast::exp2(scores[kb].thread_elements()[e] - new_max);
                probs[kb].thread_elements()[e] = half(p);
                tile_sum += p;
            }
        }
        tile_sum += simd_shuffle_xor(tile_sum, 1);
        tile_sum += simd_shuffle_xor(tile_sum, 8);
        row_sum = row_sum * factor + tile_sum;
        row_max = new_max;
        for (int db = 0; db < D / 8; ++db) {
            acc[db].thread_elements()[0] *= factor;
            acc[db].thread_elements()[1] *= factor;
            for (int kb = 0; kb < BK / 8; ++kb) {
                simdgroup_half8x8 v_frag;
                simdgroup_load(v_frag, Vs + kb * 8 * LDV + db * 8, LDV);
                simdgroup_multiply_accumulate(acc[db], probs[kb], v_frag, acc[db]);
            }
        }
    }

    if (row_ok) {
        device T* op = out + (size_t(batch * L + row) * HQ + head) * D;
        for (int db = 0; db < D / 8; ++db) {
            op[db * 8 + fn] = T(acc[db].thread_elements()[0] / row_sum);
            op[db * 8 + fn + 1] = T(acc[db].thread_elements()[1] / row_sum);
        }
    }
"""

_relative_attention_kernel = mx.fast.metal_kernel(
    name="relative_attention",
    input_names=["q", "k", "v", "rel", "sinks", "key_start", "sizes", "scale"],
    output_names=["out"],
    source=_RELATIVE_ATTENTION_SOURCE,
    ensure_row_contiguous=False,
)

_FAST_REL_ATTN = os.environ.get("LOOPLING_FAST_REL_ATTN", "1") not in ("0", "")
_REL_QSG = int(os.environ.get("LOOPLING_REL_QSG", "0"))
_REL_BK_LOAD = int(os.environ.get("LOOPLING_REL_BK_LOAD", "0"))
_REL_TRANS_K = int(os.environ.get("LOOPLING_REL_TRANS_K", "0"))
_REL_VEC_BIAS = int(os.environ.get("LOOPLING_REL_VEC_BIAS", "1"))

_RELATIVE_ATTENTION_FAST_SOURCE = """
    constexpr float LOG2E = 1.4426950408889634f;
    constexpr int BK = 16;
    constexpr int LDK = BK_LOAD + 8;
    constexpr int LDV = D + 8;
    constexpr int BQ = QSG * 8;
    constexpr int TG = QSG * 32;
    constexpr int SMEM_ELEMS = (D * LDK > BK_LOAD * LDV) ? (D * LDK) : (BK_LOAD * LDV);
    const int L = sizes[0];
    const int S = sizes[1];
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const int batch = threadgroup_position_in_grid.z;
    const int head = threadgroup_position_in_grid.y;
    const int kv_head = head / (HQ / HKV);
    const int q0 = (sizes[2] - 1 - int(threadgroup_position_in_grid.x)) * BQ;
    const int start = key_start[batch];
    const float qscale = scale * LOG2E;

    const short fm = ((lane / 4) & 4) + (lane / 2) % 4;
    const short fn = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
    const int sg_first_row = q0 + int(sg) * 8;
    const int row = sg_first_row + fm;
    const bool row_ok = row < L;
    const int diag = S - L + row;

    threadgroup half KV_smem[SMEM_ELEMS];
    threadgroup bfloat16_t rel_smem[OTF ? (QSG * 8 * 24) : 1];

    simdgroup_half8x8 q_frag[D / 8];
    #pragma clang loop unroll(full)
    for (int db = 0; db < D / 8; ++db) {
        half2 qv = half2(0.0h);
        if (row_ok) {
            const device T* qp = q + batch * q_strides[0] + head * q_strides[1]
                + row * q_strides[2] + db * 8 + fn;
            qv = half2(float(qp[0]), float(qp[1]));
        }
        q_frag[db].thread_elements()[0] = qv.x;
        q_frag[db].thread_elements()[1] = qv.y;
    }

    simdgroup_bfloat8x8 r_frag0, r_frag1;
    if constexpr (OTF) {
        const int r_safe = max(0, min(sg_first_row + int(fm), L - 1));
        const device bfloat16_t* rp0 = rel + (size_t(batch * L + r_safe) * HQ + head) * 16 + fn;
        r_frag0.thread_elements()[0] = rp0[0];
        r_frag0.thread_elements()[1] = rp0[1];
        r_frag1.thread_elements()[0] = rp0[8];
        r_frag1.thread_elements()[1] = rp0[9];
    }

    simdgroup_float8x8 acc[D / 8];
    #pragma clang loop unroll(full)
    for (int db = 0; db < D / 8; ++db) {
        acc[db] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    float row_max = row_ok ? float(sinks[(batch * L + row) * HQ + head]) * LOG2E : 0.0f;
    float row_sum = 1.0f;
    const int bias_row = batch * L + (row_ok ? row : 0);
    const device T* bias = rel + (size_t(bias_row) * HQ + head) * R;

    const int first_diag = S - L + q0;
    const int last_diag = S - L + min(q0 + BQ, L) - 1;
    int first_key = WINDOW > 0 ? max(0, first_diag - WINDOW + 1) : 0;
    first_key = max(first_key, start);
    const device T* k_head = k + batch * k_strides[0] + kv_head * k_strides[1];
    const device T* v_head = v + batch * v_strides[0] + kv_head * v_strides[1];

    const int sg_last_row = min(sg_first_row + 7, L - 1);
    const int sg_first_diag = S - L + sg_first_row;
    const int sg_last_diag = S - L + sg_last_row;
    const bool all_rows_ok = (sg_first_row + 7 < L);

    const int k_start_blk = (first_key / BK) * BK;
    for (int k_blk = k_start_blk; k_blk <= last_diag; k_blk += BK_LOAD) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int c = tid; c < BK_LOAD * D / 8; c += TG) {
            const int r = c / (D / 8);
            const int col = (c % (D / 8)) * 8;
            const int key = k_blk + r;
            half4 h0 = half4(0), h1 = half4(0);
            if (key < S) {
                const uint4 kr = *(const device uint4*)(k_head + key * k_strides[2] + col);
                h0 = half4(float4(as_type<vec<T, 4>>(kr.xy)));
                h1 = half4(float4(as_type<vec<T, 4>>(kr.zw)));
            }
            if constexpr (TRANS_K) {
                ((threadgroup half4*)(KV_smem + r * LDV + col))[0] = h0;
                ((threadgroup half4*)(KV_smem + r * LDV + col))[1] = h1;
            } else {
                KV_smem[(col + 0) * LDK + r] = h0.x;
                KV_smem[(col + 1) * LDK + r] = h0.y;
                KV_smem[(col + 2) * LDK + r] = h0.z;
                KV_smem[(col + 3) * LDK + r] = h0.w;
                KV_smem[(col + 4) * LDK + r] = h1.x;
                KV_smem[(col + 5) * LDK + r] = h1.y;
                KV_smem[(col + 6) * LDK + r] = h1.z;
                KV_smem[(col + 7) * LDK + r] = h1.w;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const bool sg_active = !(sg_first_row >= L || k_blk > sg_last_diag
            || (WINDOW > 0 && (sg_first_diag - (k_blk + BK_LOAD - 1)) >= WINDOW));

        simdgroup_float8x8 scores[BK_LOAD / 8];
        if (sg_active) {
            #pragma clang loop unroll(full)
            for (int kb = 0; kb < BK_LOAD / 8; ++kb) {
                scores[kb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            }
            #pragma clang loop unroll(full)
            for (int db = 0; db < D / 8; ++db) {
                simdgroup_half8x8 kf[BK_LOAD / 8];
                simdgroup_barrier(mem_flags::mem_none);
                #pragma clang loop unroll(full)
                for (int kb = 0; kb < BK_LOAD / 8; ++kb) {
                    if constexpr (TRANS_K) {
                        const threadgroup half* kp = KV_smem + (kb * 8 + fn) * LDV + db * 8 + fm;
                        kf[kb].thread_elements()[0] = kp[0];
                        kf[kb].thread_elements()[1] = kp[LDV];
                    } else {
                        const half2 kv = *((const threadgroup half2*)(KV_smem + (db * 8 + fm) * LDK + kb * 8 + fn));
                        kf[kb].thread_elements()[0] = kv.x;
                        kf[kb].thread_elements()[1] = kv.y;
                    }
                }
                simdgroup_barrier(mem_flags::mem_none);
                #pragma clang loop unroll(full)
                for (int kb = 0; kb < BK_LOAD / 8; ++kb) {
                    simdgroup_multiply_accumulate(scores[kb], q_frag[db], kf[kb], scores[kb]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int c = tid; c < BK_LOAD * D / 8; c += TG) {
            const int r = c / (D / 8);
            const int col = (c % (D / 8)) * 8;
            const int key = k_blk + r;
            half4 h0 = half4(0), h1 = half4(0);
            if (key < S) {
                const uint4 vr = *(const device uint4*)(v_head + key * v_strides[2] + col);
                h0 = half4(float4(as_type<vec<T, 4>>(vr.xy)));
                h1 = half4(float4(as_type<vec<T, 4>>(vr.zw)));
            }
            ((threadgroup half4*)(KV_smem + r * LDV + col))[0] = h0;
            ((threadgroup half4*)(KV_smem + r * LDV + col))[1] = h1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!sg_active) continue;

        #pragma clang loop unroll(full)
        for (int sub_i = 0; sub_i < BK_LOAD / BK; ++sub_i) {
            const int sub = sub_i * BK;
            const int k0 = k_blk + sub;
            if (k0 > sg_last_diag || (WINDOW > 0 && (sg_first_diag - (k0 + BK - 1)) >= WINDOW)) continue;
            const int kb0 = sub_i * 2;
            const int d_lo = sg_first_diag - (k0 + BK - 1);
            const int d_hi = sg_last_diag - k0;
            int d_base = 0;
            if constexpr (OTF) {
                if (d_lo < R && d_hi >= 0) {
                    d_base = max(0, min(R - 24, d_lo));
                    threadgroup bfloat16_t* rbp = rel_smem + sg * (8 * 24);
                    #pragma clang loop unroll(full)
                    for (int cb = 0; cb < 3; ++cb) {
                        const int col = d_base + cb * 8 + fn;
                        simdgroup_bfloat8x8 p0, p1;
                        const bfloat2 p0v = *(const device bfloat2*)(rel_proj + (fm + 0) * R + col);
                        p0.thread_elements()[0] = p0v.x;
                        p0.thread_elements()[1] = p0v.y;
                        const bfloat2 p1v = *(const device bfloat2*)(rel_proj + (fm + 8) * R + col);
                        p1.thread_elements()[0] = p1v.x;
                        p1.thread_elements()[1] = p1v.y;
                        simdgroup_float8x8 bc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                        simdgroup_multiply_accumulate(bc, r_frag0, p0, bc);
                        simdgroup_multiply_accumulate(bc, r_frag1, p1, bc);
                        rbp[fm * 24 + cb * 8 + fn] = bfloat16_t(bc.thread_elements()[0]);
                        rbp[fm * 24 + cb * 8 + fn + 1] = bfloat16_t(bc.thread_elements()[1]);
                    }
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
            float tile_max = -INFINITY;
            const bool full_visible = all_rows_ok && (k0 >= start) && (k0 + BK - 1 <= sg_first_diag)
                && (WINDOW == 0 || (sg_last_diag - k0) < WINDOW);
            const threadgroup bfloat16_t* rbp_row = rel_smem + sg * (8 * 24) + fm * 24 - d_base;
            if (full_visible && d_lo >= R) {
                #pragma clang loop unroll(full)
                for (int kb = 0; kb < 2; ++kb) {
                    for (int e = 0; e < 2; ++e) {
                        float s = scores[kb0 + kb].thread_elements()[e] * qscale;
                        scores[kb0 + kb].thread_elements()[e] = s;
                        tile_max = max(tile_max, s);
                    }
                }
            } else if (OTF && full_visible && d_hi < R) {
                #pragma clang loop unroll(full)
                for (int kb = 0; kb < 2; ++kb) {
                    const int d0 = diag - (k0 + kb * 8 + fn);
                    float s0 = scores[kb0 + kb].thread_elements()[0] * qscale;
                    s0 += float(rbp_row[d0]) * LOG2E;
                    float s1 = scores[kb0 + kb].thread_elements()[1] * qscale;
                    s1 += float(rbp_row[d0 - 1]) * LOG2E;
                    scores[kb0 + kb].thread_elements()[0] = s0;
                    scores[kb0 + kb].thread_elements()[1] = s1;
                    tile_max = max(tile_max, max(s0, s1));
                }
            } else if (!OTF && VEC_BIAS && full_visible && d_hi < R) {
                #pragma clang loop unroll(full)
                for (int kb = 0; kb < 2; ++kb) {
                    const int d0 = diag - (k0 + kb * 8 + fn);
                    const packed_ushort2 raw = *(const device packed_ushort2*)(bias + d0 - 1);
                    float s0 = scores[kb0 + kb].thread_elements()[0] * qscale;
                    s0 += float(as_type<T>(raw.y)) * LOG2E;
                    float s1 = scores[kb0 + kb].thread_elements()[1] * qscale;
                    s1 += float(as_type<T>(raw.x)) * LOG2E;
                    scores[kb0 + kb].thread_elements()[0] = s0;
                    scores[kb0 + kb].thread_elements()[1] = s1;
                    tile_max = max(tile_max, s0);
                    tile_max = max(tile_max, s1);
                }
            } else {
                #pragma clang loop unroll(full)
                for (int kb = 0; kb < 2; ++kb) {
                    for (int e = 0; e < 2; ++e) {
                        const int key = k0 + kb * 8 + fn + e;
                        const int d = diag - key;
                        const bool vis = row_ok && d >= 0 && key >= start && (WINDOW == 0 || d < WINDOW);
                        float s = -INFINITY;
                        if (vis) {
                            s = scores[kb0 + kb].thread_elements()[e] * qscale;
                            if (d < R) {
                                s += (OTF ? float(rbp_row[d]) : float(bias[d])) * LOG2E;
                            }
                        }
                        scores[kb0 + kb].thread_elements()[e] = s;
                        tile_max = max(tile_max, s);
                    }
                }
            }
            tile_max = max(tile_max, simd_shuffle_xor(tile_max, 1));
            tile_max = max(tile_max, simd_shuffle_xor(tile_max, 8));
            const float new_max = max(row_max, tile_max);
            const float factor = fast::exp2(row_max - new_max);
            const bool skip_pv = simd_all(tile_max - new_max < -25.0f);
            float tile_sum = 0.0f;
            simdgroup_half8x8 probs[2];
            #pragma clang loop unroll(full)
            for (int kb = 0; kb < 2; ++kb) {
                for (int e = 0; e < 2; ++e) {
                    const float p = fast::exp2(scores[kb0 + kb].thread_elements()[e] - new_max);
                    probs[kb].thread_elements()[e] = half(p);
                    tile_sum += p;
                }
            }
            tile_sum += simd_shuffle_xor(tile_sum, 1);
            tile_sum += simd_shuffle_xor(tile_sum, 8);
            row_sum = row_sum * factor + tile_sum;
            row_max = new_max;
            if (factor != 1.0f) {
                #pragma clang loop unroll(full)
                for (int db = 0; db < D / 8; ++db) {
                    acc[db].thread_elements()[0] *= factor;
                    acc[db].thread_elements()[1] *= factor;
                }
            }
            if (!skip_pv) {
                const threadgroup half* Vc = KV_smem + sub * LDV;
                #pragma clang loop unroll(full)
                for (int kb = 0; kb < 2; ++kb) {
                    #pragma clang loop unroll(full)
                    for (int db0 = 0; db0 < D / 8; db0 += 4) {
                        simdgroup_half8x8 vf[4];
                        simdgroup_barrier(mem_flags::mem_none);
                        #pragma clang loop unroll(full)
                        for (int j = 0; j < 4; ++j) {
                            const half2 vv = *((const threadgroup half2*)(Vc + (kb * 8 + fm) * LDV + (db0 + j) * 8 + fn));
                            vf[j].thread_elements()[0] = vv.x;
                            vf[j].thread_elements()[1] = vv.y;
                        }
                        simdgroup_barrier(mem_flags::mem_none);
                        #pragma clang loop unroll(full)
                        for (int j = 0; j < 4; ++j) {
                            simdgroup_multiply_accumulate(acc[db0 + j], probs[kb], vf[j], acc[db0 + j]);
                        }
                    }
                }
            }
        }
    }

    if (row_ok) {
        device T* op = out + (size_t(batch * L + row) * HQ + head) * D;
        #pragma clang loop unroll(full)
        for (int db = 0; db < D / 8; ++db) {
            op[db * 8 + fn] = T(acc[db].thread_elements()[0] / row_sum);
            op[db * 8 + fn + 1] = T(acc[db].thread_elements()[1] / row_sum);
        }
    }
"""

_relative_attention_fast_kernel = mx.fast.metal_kernel(
    name="relative_attention_fast",
    input_names=["q", "k", "v", "rel", "sinks", "key_start", "sizes", "scale", "rel_proj"],
    output_names=["out"],
    source=_RELATIVE_ATTENTION_FAST_SOURCE,
    ensure_row_contiguous=False,
)
_DUMMY_REL_PROJ = mx.zeros((16, 24), dtype=mx.bfloat16)


def relative_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    rel: mx.array,
    sinks: mx.array,
    *,
    scale: float,
    window: int,
    key_start: mx.array,
    rel_proj: mx.array | None = None,
) -> mx.array:
    """q: (B, Hq, L, D), k and v: (B, Hkv, S, D), all bfloat16, with the queries
    at the last L keys. rel: (B, L, Hq, R) position bias, OR when rel_proj is passed
    (B, L, Hq, 16) clipped relative_states with rel_proj (16, R) bf16 for OTF bias."""
    B, Hq, L, D = q.shape
    Hkv, S = k.shape[1], k.shape[2]
    otf = rel_proj is not None and rel.shape[-1] == 16 and rel_proj.shape[1] >= 24
    if _FAST_REL_ATTN and Hq % Hkv == 0:
        qsg = _REL_QSG if _REL_QSG > 0 else (8 if L >= 768 else 4)
        bkl = _REL_BK_LOAD if _REL_BK_LOAD > 0 else (64 if L >= 768 else 16)
        bq = qsg * 8
        nq_tiles = (L + bq - 1) // bq
        tg = qsg * 32
        r_ext = rel_proj.shape[1] if otf else rel.shape[-1]
        r_proj_in = rel_proj if otf else _DUMMY_REL_PROJ
        return _relative_attention_fast_kernel(
            inputs=[
                q,
                k,
                v,
                mx.contiguous(rel),
                mx.contiguous(sinks.astype(mx.float32)),
                mx.contiguous(key_start.astype(mx.int32)),
                mx.array([L, S, nq_tiles], dtype=mx.int32),
                scalar(scale),
                r_proj_in,
            ],
            template=[
                ("T", q.dtype),
                ("D", D),
                ("HQ", Hq),
                ("HKV", Hkv),
                ("R", r_ext),
                ("WINDOW", window),
                ("QSG", qsg),
                ("BK_LOAD", bkl),
                ("TRANS_K", _REL_TRANS_K),
                ("VEC_BIAS", _REL_VEC_BIAS),
                ("OTF", int(otf)),
            ],
            grid=(nq_tiles * tg, Hq, B),
            threadgroup=(tg, 1, 1),
            output_shapes=[(B, L, Hq, D)],
            output_dtypes=[q.dtype],
        )[0]
    if otf:
        rel = rel @ rel_proj
    return _relative_attention_kernel(
        inputs=[
            q,
            k,
            v,
            mx.contiguous(rel),
            mx.contiguous(sinks.astype(mx.float32)),
            mx.contiguous(key_start.astype(mx.int32)),
            mx.array([L, S], dtype=mx.int32),
            scalar(scale),
        ],
        template=[
            ("T", q.dtype),
            ("D", D),
            ("HQ", Hq),
            ("HKV", Hkv),
            ("R", rel.shape[-1]),
            ("WINDOW", window),
            ("BQ", _BQ),
            ("TG", _TG),
        ],
        grid=((L + _BQ - 1) // _BQ * _TG, Hq, B),
        threadgroup=(_TG, 1, 1),
        output_shapes=[(B, L, Hq, D)],
        output_dtypes=[q.dtype],
    )[0]
