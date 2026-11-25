"""Fast multi-row 4-bit affine quantized matvec and fused 4-bit Markov bias+argmax/topk kernels for speculative drafting."""
import os

import mlx.core as mx

from .glue import flit

MAX_B = 16
_GS = 64
_BITS = 4

_QMV4_SRC = r"""
  constexpr int group_size = 64;
  constexpr int bits = 4;
  constexpr int values_per_thread = 8;
  constexpr int block_size = values_per_thread * 32;  // 256
  constexpr int scale_step = group_size / values_per_thread;  // 8 lanes per group
  static_assert(K % block_size == 0, "qmv4 needs K % 256 == 0");
  constexpr bool FULL = (B % RB) == 0;

  uint ty = threadgroup_position_in_grid.y;
  uint b0 = threadgroup_position_in_grid.z * RB;
  uint simd_gid = simdgroup_index_in_threadgroup;
  uint simd_lid = thread_index_in_simdgroup;

  const int out_row = int(ty) * (NSG * RPS) + int(simd_gid) * RPS;
  int last_row = RPS - 1;
  if (PR) {
    if (out_row >= N) return;
    last_row = min(last_row, N - 1 - out_row);
  }

  const int nb = FULL ? RB : min(RB, B - int(b0));
  typedef float U;
  thread U result[RB][RPS];
  for (int b = 0; b < RB; b++)
    for (int r = 0; r < RPS; r++) result[b][r] = 0;

  const int in_vec_size_w = K / 2;   // 4-bit: 2 values per byte
  const int in_vec_size_g = K / group_size;

  const device uint8_t* ws = (const device uint8_t*)w + size_t(out_row) * in_vec_size_w + simd_lid * 4;
  const device bfloat16_t* sc_p = scales + size_t(out_row) * in_vec_size_g + (simd_lid >> 3);
  const device bfloat16_t* bi_p = biases + size_t(out_row) * in_vec_size_g + (simd_lid >> 3);
  const device bfloat16_t* xp0 = x + size_t(b0) * K + simd_lid * values_per_thread;
  device bfloat16_t* yp = y + size_t(b0) * N + out_row;

  for (int k = 0; k < K; k += block_size) {
    thread uint32_t wr[RPS];
    thread U sc[RPS];
    thread U bi[RPS];
    _Pragma("clang loop unroll(full)")
    for (int row = 0; row < RPS; row++) {
      int src = PR ? min(row, last_row) : row;
      wr[row] = ((const device uint32_t*)(ws + size_t(src) * in_vec_size_w))[0];
      sc[row] = U(sc_p[size_t(src) * in_vec_size_g]);
      bi[row] = U(bi_p[size_t(src) * in_vec_size_g]);
    }
    _Pragma("clang loop unroll(full)")
    for (int b = 0; b < RB; b++) {
      if (FULL || b < nb) {
        uint4 xpk = ((const device uint4*)(xp0 + size_t(b) * K))[0];
        float4 xv0 = float4(as_type<bfloat4>(xpk.xy));
        float4 xv1 = float4(as_type<bfloat4>(xpk.zw));
        U smb = (xv0.x + xv0.y + xv0.z + xv0.w) + (xv1.x + xv1.y + xv1.z + xv1.w);
        _Pragma("clang loop unroll(full)")
        for (int row = 0; row < RPS; row++) {
          uint32_t wv = wr[row];
          U accum = (xv0.x * float(wv & 0xFu)
                   + xv0.y * float((wv >> 4) & 0xFu)
                   + xv0.z * float((wv >> 8) & 0xFu)
                   + xv0.w * float((wv >> 12) & 0xFu))
                  + (xv1.x * float((wv >> 16) & 0xFu)
                   + xv1.y * float((wv >> 20) & 0xFu)
                   + xv1.z * float((wv >> 24) & 0xFu)
                   + xv1.w * float((wv >> 28) & 0xFu));
          result[b][row] += sc[row] * accum + smb * bi[row];
        }
      }
    }
    ws += 128;      // 256 / 2 bytes
    sc_p += 4;      // 256 / 64 groups
    bi_p += 4;
    xp0 += block_size;
  }

  _Pragma("clang loop unroll(full)")
  for (int b = 0; b < RB; b++) {
    _Pragma("clang loop unroll(full)")
    for (int row = 0; row < RPS; row++) {
      U r = simd_sum(result[b][row]);
      if (simd_lid == 0 && row <= last_row && (FULL || b < nb)) {
        yp[size_t(b) * N + row] = static_cast<bfloat16_t>(r);
      }
    }
  }
"""

_MARKOV_P1_SRC = r"""
  uint sp = threadgroup_position_in_grid.x;
  uint lid = thread_position_in_threadgroup.x;
  uint lane = lid & 31, sg = lid >> 5;

  // latent is (1, 256) bf16: each lane holds 8 elements in [lane*8 .. lane*8+7], all inside group (lane>>3)
  uint4 lpk = ((const device uint4*)(latent + lane * 8))[0];
  float4 lv0 = float4(as_type<bfloat4>(lpk.xy));
  float4 lv1 = float4(as_type<bfloat4>(lpk.zw));
  float l_sum = (lv0.x + lv0.y + lv0.z + lv0.w) + (lv1.x + lv1.y + lv1.z + lv1.w);

  const int chunk = (VV + SP - 1) / SP;
  const int r0 = int(sp) * chunk;
  const int r1 = min(VV, r0 + chunk);

  float bval = -INFINITY;
  uint bidx = 0xFFFFFFFFu;
  const uint g = lane >> 3;

  int r = r0 + int(sg);
  for (; r + NSG < r1; r += 2 * NSG) {
    int rr = r + NSG;
    uint32_t wr0 = fw_w[size_t(r) * 32 + lane];
    float sc0 = float(fw_s[size_t(r) * 4 + g]);
    float bi0 = float(fw_b[size_t(r) * 4 + g]);
    uint32_t wr1 = fw_w[size_t(rr) * 32 + lane];
    float sc1 = float(fw_s[size_t(rr) * 4 + g]);
    float bi1 = float(fw_b[size_t(rr) * 4 + g]);
    float dot0 = (lv0.x * float(wr0 & 0xFu)
                + lv0.y * float((wr0 >> 4) & 0xFu)
                + lv0.z * float((wr0 >> 8) & 0xFu)
                + lv0.w * float((wr0 >> 12) & 0xFu))
               + (lv1.x * float((wr0 >> 16) & 0xFu)
                + lv1.y * float((wr0 >> 20) & 0xFu)
                + lv1.z * float((wr0 >> 24) & 0xFu)
                + lv1.w * float((wr0 >> 28) & 0xFu));
    float dot1 = (lv0.x * float(wr1 & 0xFu)
                + lv0.y * float((wr1 >> 4) & 0xFu)
                + lv0.z * float((wr1 >> 8) & 0xFu)
                + lv0.w * float((wr1 >> 12) & 0xFu))
               + (lv1.x * float((wr1 >> 16) & 0xFu)
                + lv1.y * float((wr1 >> 20) & 0xFu)
                + lv1.z * float((wr1 >> 24) & 0xFu)
                + lv1.w * float((wr1 >> 28) & 0xFu));
    float val0 = simd_sum(sc0 * dot0 + l_sum * bi0);
    float val1 = simd_sum(sc1 * dot1 + l_sum * bi1);
    if (lane == 0) {
      float t0 = float(base_row[r]) + float(fold_b[r]) + val0;
      if (t0 > bval || (t0 == bval && uint(r) < bidx)) {
        bval = t0;
        bidx = uint(r);
      }
      float t1 = float(base_row[rr]) + float(fold_b[rr]) + val1;
      if (t1 > bval || (t1 == bval && uint(rr) < bidx)) {
        bval = t1;
        bidx = uint(rr);
      }
    }
  }
  if (r < r1) {
    uint32_t wr = fw_w[size_t(r) * 32 + lane];
    float sc = float(fw_s[size_t(r) * 4 + g]);
    float bi = float(fw_b[size_t(r) * 4 + g]);
    float dot = (lv0.x * float(wr & 0xFu)
               + lv0.y * float((wr >> 4) & 0xFu)
               + lv0.z * float((wr >> 8) & 0xFu)
               + lv0.w * float((wr >> 12) & 0xFu))
              + (lv1.x * float((wr >> 16) & 0xFu)
               + lv1.y * float((wr >> 20) & 0xFu)
               + lv1.z * float((wr >> 24) & 0xFu)
               + lv1.w * float((wr >> 28) & 0xFu));
    float val = simd_sum(sc * dot + l_sum * bi);
    if (lane == 0) {
      float tot = float(base_row[r]) + float(fold_b[r]) + val;
      if (tot > bval || (tot == bval && uint(r) < bidx)) {
        bval = tot;
        bidx = uint(r);
      }
    }
  }

  threadgroup float sv[NSG];
  threadgroup uint si[NSG];
  if (lane == 0) {
    sv[sg] = bval;
    si[sg] = bidx;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float fv = -INFINITY;
    uint fi = 0xFFFFFFFFu;
    if (lane < NSG) {
      fv = sv[lane];
      fi = si[lane];
    }
    for (uint d = 16; d > 0; d >>= 1) {
      float ov = simd_shuffle_down(fv, d);
      uint oi = simd_shuffle_down(fi, d);
      if (ov > fv || (ov == fv && oi < fi)) {
        fv = ov;
        fi = oi;
      }
    }
    if (lane == 0) {
      pval[sp] = fv;
      pidx[sp] = fi;
    }
  }
"""

_MARKOV_P2_SRC = r"""
  uint lid = thread_position_in_threadgroup.x;
  uint lane = lid & 31, sg = lid >> 5;
  float fv = -INFINITY;
  uint fi = 0xFFFFFFFFu;
  for (int i = int(lid); i < SP; i += 128) {
    float ov = pval[i];
    uint oi = pidx[i];
    if (ov > fv || (ov == fv && oi < fi)) {
      fv = ov;
      fi = oi;
    }
  }
  for (uint d = 16; d > 0; d >>= 1) {
    float ov = simd_shuffle_down(fv, d);
    uint oi = simd_shuffle_down(fi, d);
    if (ov > fv || (ov == fv && oi < fi)) {
      fv = ov;
      fi = oi;
    }
  }
  threadgroup float sv[4];
  threadgroup uint si[4];
  if (lane == 0) { sv[sg] = fv; si[sg] = fi; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float bf = -INFINITY;
    uint bi = 0xFFFFFFFFu;
    if (lane < 4) { bf = sv[lane]; bi = si[lane]; }
    for (uint d = 2; d > 0; d >>= 1) {
      float ov = simd_shuffle_down(bf, d);
      uint oi = simd_shuffle_down(bi, d);
      if (ov > bf || (ov == bf && oi < bi)) { bf = ov; bi = oi; }
    }
    if (lane == 0) out[0] = bi;
  }
"""

_QMV4_RO_SRC = r"""
  constexpr int group_size = 64;
  constexpr int block_size = 256;
  static_assert(K % block_size == 0, "qmv4 needs K % 256 == 0");
  constexpr bool FULL = (B % RB) == 0;

  uint ty = threadgroup_position_in_grid.y;
  uint b0 = threadgroup_position_in_grid.z * RB;
  uint simd_gid = simdgroup_index_in_threadgroup;
  uint simd_lid = thread_index_in_simdgroup;

  const int out_row = int(ty) * (NSG * RPS) + int(simd_gid) * RPS;
  int last_row = RPS - 1;
  if (PR) {
    if (out_row >= N) return;
    last_row = min(last_row, N - 1 - out_row);
  }

  const int nb = FULL ? RB : min(RB, B - int(b0));
  thread float result[RB][RPS];
  for (int b = 0; b < RB; b++)
    for (int r = 0; r < RPS; r++) result[b][r] = 0.0f;

  const int in_vec_size_w = K / 2;
  const int in_vec_size_g = K / group_size;

  const device uint8_t* ws = (const device uint8_t*)w + size_t(out_row) * in_vec_size_w + simd_lid * 4;
  const device bfloat16_t* sc_p = scales + size_t(out_row) * in_vec_size_g + (simd_lid >> 3);
  const device bfloat16_t* bi_p = biases + size_t(out_row) * in_vec_size_g + (simd_lid >> 3);
  const device bfloat16_t* xp0 = x + size_t(b0) * K + simd_lid * 8;
  device bfloat16_t* yp = y + size_t(b0) * N + out_row;

  for (int k = 0; k < K; k += block_size) {
    _Pragma("clang loop unroll(full)")
    for (int row = 0; row < RPS; row++) {
      int src = PR ? min(row, last_row) : row;
      uint32_t wv = ((const device uint32_t*)(ws + size_t(src) * in_vec_size_w))[0];
      float sc = float(sc_p[size_t(src) * in_vec_size_g]);
      float bi = float(bi_p[size_t(src) * in_vec_size_g]);
      float4 w0 = float4(float(wv & 0xFu), float((wv >> 4) & 0xFu), float((wv >> 8) & 0xFu), float((wv >> 12) & 0xFu)) * sc + bi;
      float4 w1 = float4(float((wv >> 16) & 0xFu), float((wv >> 20) & 0xFu), float((wv >> 24) & 0xFu), float((wv >> 28) & 0xFu)) * sc + bi;
      _Pragma("clang loop unroll(full)")
      for (int b = 0; b < RB; b++) {
        if (FULL || b < nb) {
          uint4 xpk = ((const device uint4*)(xp0 + size_t(b) * K))[0];
          float4 xv0 = float4(as_type<bfloat4>(xpk.xy));
          float4 xv1 = float4(as_type<bfloat4>(xpk.zw));
          result[b][row] += dot(xv0, w0) + dot(xv1, w1);
        }
      }
    }
    ws += 128;
    sc_p += 4;
    bi_p += 4;
    xp0 += block_size;
  }

  _Pragma("clang loop unroll(full)")
  for (int b = 0; b < RB; b++) {
    _Pragma("clang loop unroll(full)")
    for (int row = 0; row < RPS; row++) {
      float r = simd_sum(result[b][row]);
      if (simd_lid == 0 && row <= last_row && (FULL || b < nb)) {
        yp[size_t(b) * N + row] = static_cast<bfloat16_t>(r);
      }
    }
  }
"""

def _markov_topk4_p1_src(scap_lit):
    return f"""
  uint sp = threadgroup_position_in_grid.x;
  uint lid = thread_position_in_threadgroup.x;
  uint lane = lid & 31, sg = lid >> 5;

  float4 lv0[MM], lv1[MM];
  for (int m = 0; m < MM; m++) {{
    uint4 lpk = ((const device uint4*)(latent + size_t(m) * 256 + lane * 8))[0];
    lv0[m] = float4(as_type<bfloat4>(lpk.xy));
    lv1[m] = float4(as_type<bfloat4>(lpk.zw));
  }}

  const int chunk = (VV + SP - 1) / SP;
  const int r0 = int(sp) * chunk;
  const int r1 = min(VV, r0 + chunk);
  const uint g = lane >> 3;

  float tv[MM][4];
  uint ti[MM][4];
  float smax[MM], ssum[MM];
  for (int m = 0; m < MM; m++) {{
    for (int k = 0; k < 4; k++) {{ tv[m][k] = -INFINITY; ti[m][k] = 0xFFFFFFFFu; }}
    smax[m] = -INFINITY; ssum[m] = 0.0f;
  }}

  const float sc_cap = {scap_lit};
  const float inv_cap = 1.0f / {scap_lit};

  auto push_row = [&](int r, int m, float val) {{
    if (lane == 0) {{
      float lg = sc_cap * fast::tanh((float(base_row[r]) + float(fold_b[r]) + val) * inv_cap);
      float nm = max(smax[m], lg);
      ssum[m] = (smax[m] > -INFINITY ? ssum[m] * fast::exp(smax[m] - nm) : 0.0f) + fast::exp(lg - nm);
      smax[m] = nm;
      if (lg > tv[m][3] || (lg == tv[m][3] && uint(r) < ti[m][3])) {{
        int pos = 3;
        while (pos > 0 && (lg > tv[m][pos - 1] || (lg == tv[m][pos - 1] && uint(r) < ti[m][pos - 1]))) {{
          tv[m][pos] = tv[m][pos - 1];
          ti[m][pos] = ti[m][pos - 1];
          pos--;
        }}
        tv[m][pos] = lg;
        ti[m][pos] = uint(r);
      }}
    }}
  }};

  int r = r0 + int(sg);
  for (; r + NSG < r1; r += 2 * NSG) {{
    int rr = r + NSG;
    uint32_t wr0 = fw_w[size_t(r) * 32 + lane];
    float sc0 = float(fw_s[size_t(r) * 4 + g]);
    float bi0 = float(fw_b[size_t(r) * 4 + g]);
    float4 w00 = float4(float(wr0 & 0xFu), float((wr0 >> 4) & 0xFu), float((wr0 >> 8) & 0xFu), float((wr0 >> 12) & 0xFu)) * sc0 + bi0;
    float4 w01 = float4(float((wr0 >> 16) & 0xFu), float((wr0 >> 20) & 0xFu), float((wr0 >> 24) & 0xFu), float((wr0 >> 28) & 0xFu)) * sc0 + bi0;

    uint32_t wr1 = fw_w[size_t(rr) * 32 + lane];
    float sc1 = float(fw_s[size_t(rr) * 4 + g]);
    float bi1 = float(fw_b[size_t(rr) * 4 + g]);
    float4 w10 = float4(float(wr1 & 0xFu), float((wr1 >> 4) & 0xFu), float((wr1 >> 8) & 0xFu), float((wr1 >> 12) & 0xFu)) * sc1 + bi1;
    float4 w11 = float4(float((wr1 >> 16) & 0xFu), float((wr1 >> 20) & 0xFu), float((wr1 >> 24) & 0xFu), float((wr1 >> 28) & 0xFu)) * sc1 + bi1;

    float base0 = float(base_row[r]) + float(fold_b[r]);
    float base1 = float(base_row[rr]) + float(fold_b[rr]);
    for (int m = 0; m < MM; m++) {{
      float val0 = simd_sum(dot(lv0[m], w00) + dot(lv1[m], w01));
      float val1 = simd_sum(dot(lv0[m], w10) + dot(lv1[m], w11));
      if (lane == 0) {{
        float lg0 = sc_cap * fast::tanh((base0 + val0) * inv_cap);
        float lg1 = sc_cap * fast::tanh((base1 + val1) * inv_cap);
        float nm = max(smax[m], max(lg0, lg1));
        ssum[m] = (smax[m] > -INFINITY ? ssum[m] * fast::exp(smax[m] - nm) : 0.0f)
                + fast::exp(lg0 - nm) + fast::exp(lg1 - nm);
        smax[m] = nm;
        if (lg0 > tv[m][3] || (lg0 == tv[m][3] && uint(r) < ti[m][3])) {{
          int pos = 3;
          while (pos > 0 && (lg0 > tv[m][pos - 1] || (lg0 == tv[m][pos - 1] && uint(r) < ti[m][pos - 1]))) {{
            tv[m][pos] = tv[m][pos - 1]; ti[m][pos] = ti[m][pos - 1]; pos--;
          }}
          tv[m][pos] = lg0; ti[m][pos] = uint(r);
        }}
        if (lg1 > tv[m][3] || (lg1 == tv[m][3] && uint(rr) < ti[m][3])) {{
          int pos = 3;
          while (pos > 0 && (lg1 > tv[m][pos - 1] || (lg1 == tv[m][pos - 1] && uint(rr) < ti[m][pos - 1]))) {{
            tv[m][pos] = tv[m][pos - 1]; ti[m][pos] = ti[m][pos - 1]; pos--;
          }}
          tv[m][pos] = lg1; ti[m][pos] = uint(rr);
        }}
      }}
    }}
  }}
  if (r < r1) {{
    uint32_t wr = fw_w[size_t(r) * 32 + lane];
    float sc = float(fw_s[size_t(r) * 4 + g]);
    float bi = float(fw_b[size_t(r) * 4 + g]);
    float4 w0 = float4(float(wr & 0xFu), float((wr >> 4) & 0xFu), float((wr >> 8) & 0xFu), float((wr >> 12) & 0xFu)) * sc + bi;
    float4 w1 = float4(float((wr >> 16) & 0xFu), float((wr >> 20) & 0xFu), float((wr >> 24) & 0xFu), float((wr >> 28) & 0xFu)) * sc + bi;
    for (int m = 0; m < MM; m++) {{
      float val = simd_sum(dot(lv0[m], w0) + dot(lv1[m], w1));
      push_row(r, m, val);
    }}
  }}

  threadgroup float tgv[NSG * 4];
  threadgroup uint tgi[NSG * 4];
  threadgroup float tgm[NSG];
  threadgroup float tgs[NSG];
  for (int m = 0; m < MM; m++) {{
    if (lane == 0) {{
      for (int k = 0; k < 4; k++) {{ tgv[sg * 4 + k] = tv[m][k]; tgi[sg * 4 + k] = ti[m][k]; }}
      tgm[sg] = smax[m]; tgs[sg] = ssum[m];
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane == 0) {{
      float fm = -INFINITY, fs = 0.0f;
      for (int s = 0; s < NSG; s++) {{
        float om = tgm[s];
        if (om > -INFINITY) {{
          float nm = max(fm, om);
          fs = (fm > -INFINITY ? fs * fast::exp(fm - nm) : 0.0f) + tgs[s] * fast::exp(om - nm);
          fm = nm;
        }}
      }}
      pmax[size_t(m) * SP + sp] = fm;
      psum[size_t(m) * SP + sp] = fs;
      float bv[4] = {{-INFINITY, -INFINITY, -INFINITY, -INFINITY}};
      uint bi4[4] = {{0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu}};
      for (int j = 0; j < NSG * 4; j++) {{
        float v = tgv[j]; uint idx = tgi[j];
        if (v > bv[3] || (v == bv[3] && idx < bi4[3])) {{
          int pos = 3;
          while (pos > 0 && (v > bv[pos - 1] || (v == bv[pos - 1] && idx < bi4[pos - 1]))) {{
            bv[pos] = bv[pos - 1]; bi4[pos] = bi4[pos - 1]; pos--;
          }}
          bv[pos] = v; bi4[pos] = idx;
        }}
      }}
      for (int k = 0; k < 4; k++) {{
        pval[(size_t(m) * SP + sp) * 4 + k] = bv[k];
        pidx[(size_t(m) * SP + sp) * 4 + k] = bi4[k];
      }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }}
"""

_MARKOV_TOPK4_P2_SRC = r"""
  uint m = threadgroup_position_in_grid.x;
  uint lid = thread_position_in_threadgroup.x;
  uint lane = lid & 31, sg = lid >> 5;

  float fm = -INFINITY, fs = 0.0f;
  float bv[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
  uint bi4[4] = {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu};

  for (int i = int(lid); i < SP; i += 128) {
    size_t off = size_t(m) * SP + i;
    float om = pmax[off];
    if (om > -INFINITY) {
      float os = psum[off];
      float nm = max(fm, om);
      fs = (fm > -INFINITY ? fs * fast::exp(fm - nm) : 0.0f) + os * fast::exp(om - nm);
      fm = nm;
    }
    for (int k = 0; k < 4; k++) {
      float v = pval[off * 4 + k];
      uint idx = pidx[off * 4 + k];
      if (v > bv[3] || (v == bv[3] && idx < bi4[3])) {
        int pos = 3;
        while (pos > 0 && (v > bv[pos - 1] || (v == bv[pos - 1] && idx < bi4[pos - 1]))) {
          bv[pos] = bv[pos - 1]; bi4[pos] = bi4[pos - 1]; pos--;
        }
        bv[pos] = v; bi4[pos] = idx;
      }
    }
  }

  for (uint d = 16; d > 0; d >>= 1) {
    float om = simd_shuffle_down(fm, d);
    float os = simd_shuffle_down(fs, d);
    float ovs[4] = {
      simd_shuffle_down(bv[0], d), simd_shuffle_down(bv[1], d),
      simd_shuffle_down(bv[2], d), simd_shuffle_down(bv[3], d),
    };
    uint ois[4] = {
      simd_shuffle_down(bi4[0], d), simd_shuffle_down(bi4[1], d),
      simd_shuffle_down(bi4[2], d), simd_shuffle_down(bi4[3], d),
    };
    if (om > -INFINITY) {
      float nm = max(fm, om);
      fs = (fm > -INFINITY ? fs * fast::exp(fm - nm) : 0.0f) + os * fast::exp(om - nm);
      fm = nm;
    }
    for (int k = 0; k < 4; k++) {
      float ov = ovs[k];
      uint oi = ois[k];
      if (ov > bv[3] || (ov == bv[3] && oi < bi4[3])) {
        int pos = 3;
        while (pos > 0 && (ov > bv[pos - 1] || (ov == bv[pos - 1] && oi < bi4[pos - 1]))) {
          bv[pos] = bv[pos - 1]; bi4[pos] = bi4[pos - 1]; pos--;
        }
        bv[pos] = ov; bi4[pos] = oi;
      }
    }
  }

  threadgroup float sm[4], ss[4], sv[16];
  threadgroup uint si[16];
  if (lane == 0) {
    sm[sg] = fm;
    ss[sg] = fs;
    for (int k = 0; k < 4; k++) { sv[sg * 4 + k] = bv[k]; si[sg * 4 + k] = bi4[k]; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0 && lane == 0) {
    float gm = -INFINITY, gs = 0.0f;
    for (int s = 0; s < 4; s++) {
      float om = sm[s];
      if (om > -INFINITY) {
        float nm = max(gm, om);
        gs = (gm > -INFINITY ? gs * fast::exp(gm - nm) : 0.0f) + ss[s] * fast::exp(om - nm);
        gm = nm;
      }
    }
    float lse = gm + metal::log(gs);
    float fbv[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    uint fbi[4] = {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu};
    for (int j = 0; j < 16; j++) {
      float v = sv[j]; uint idx = si[j];
      if (v > fbv[3] || (v == fbv[3] && idx < fbi[3])) {
        int pos = 3;
        while (pos > 0 && (v > fbv[pos - 1] || (v == fbv[pos - 1] && idx < fbi[pos - 1]))) {
          fbv[pos] = fbv[pos - 1]; fbi[pos] = fbi[pos - 1]; pos--;
        }
        fbv[pos] = v; fbi[pos] = idx;
      }
    }
    for (int k = 0; k < 4; k++) {
      out_tok[m * 4 + k] = fbi[k];
      out_prob[m * 4 + k] = (fbv[k] > -INFINITY) ? fast::exp(fbv[k] - lse) : 0.0f;
    }
  }
"""

def _qmv4_fp16_src(swiglu=False):
    if not swiglu:
        epi = r"""
  _Pragma("clang loop unroll(full)")
  for (int b = 0; b < RB; b++) {
    _Pragma("clang loop unroll(full)")
    for (int row = 0; row < RPS; row++) {
      U r = simd_sum(result[b][row]);
      if (simd_lid == 0 && row <= last_row && (FULL || b < nb)) {
        yp[size_t(b) * N + row] = static_cast<half>(r);
      }
    }
  }
"""
        y_ptr = "  device half* yp = y + size_t(b0) * N + out_row;"
    else:
        epi = r"""
  _Pragma("clang loop unroll(full)")
  for (int b = 0; b < RB; b++) {
    _Pragma("clang loop unroll(full)")
    for (int pair = 0; pair < RPS / 2; pair++) {
      U rg = simd_sum(result[b][2 * pair]);
      U ru = simd_sum(result[b][2 * pair + 1]);
      if (simd_lid == 0 && (2 * pair + 1) <= last_row && (FULL || b < nb)) {
        float g = float(static_cast<half>(rg));
        float u = float(static_cast<half>(ru));
        float act = (g / (1.0f + fast::exp(-g))) * u;
        yp[size_t(b) * (N / 2) + pair] = static_cast<half>(act);
      }
    }
  }
"""
        y_ptr = "  device half* yp = y + size_t(b0) * (N / 2) + (out_row / 2);"
    return r"""
  constexpr int group_size = 64;
  constexpr int bits = 4;
  constexpr int values_per_thread = 8;
  constexpr int block_size = values_per_thread * 32;  // 256
  static_assert(K % block_size == 0, "qmv4 fp16 needs K % 256 == 0");
  constexpr bool FULL = (B % RB) == 0;

  uint ty = threadgroup_position_in_grid.y;
  uint b0 = threadgroup_position_in_grid.z * RB;
  uint simd_gid = simdgroup_index_in_threadgroup;
  uint simd_lid = thread_index_in_simdgroup;

  const int out_row = int(ty) * (NSG * RPS) + int(simd_gid) * RPS;
  int last_row = RPS - 1;
  if (PR) {
    if (out_row >= N) return;
    last_row = min(last_row, N - 1 - out_row);
  }

  const int nb = FULL ? RB : min(RB, B - int(b0));
  typedef float U;
  thread U result[RB][RPS];
  for (int b = 0; b < RB; b++)
    for (int r = 0; r < RPS; r++) result[b][r] = 0;

  const int in_vec_size_w = K / 2;
  const int in_vec_size_g = K / group_size;

  const device uint8_t* ws = (const device uint8_t*)w + size_t(out_row) * in_vec_size_w + simd_lid * 4;
  const device half* sc_p = scales + size_t(out_row) * in_vec_size_g + (simd_lid >> 3);
  const device half* bi_p = biases + size_t(out_row) * in_vec_size_g + (simd_lid >> 3);
  const device half* xp0 = x + size_t(b0) * K + simd_lid * values_per_thread;
""" + y_ptr + r"""

  for (int k = 0; k < K; k += block_size) {
    thread float4 w0[RPS];
    thread float4 w1[RPS];
    thread U sc[RPS];
    thread U bi[RPS];
    _Pragma("clang loop unroll(full)")
    for (int row = 0; row < RPS; row++) {
      int src = PR ? min(row, last_row) : row;
      uint32_t wv = ((const device uint32_t*)(ws + size_t(src) * in_vec_size_w))[0];
      w0[row] = float4(float(wv & 0xFu), float((wv >> 4) & 0xFu), float((wv >> 8) & 0xFu), float((wv >> 12) & 0xFu));
      w1[row] = float4(float((wv >> 16) & 0xFu), float((wv >> 20) & 0xFu), float((wv >> 24) & 0xFu), float((wv >> 28) & 0xFu));
      sc[row] = U(sc_p[size_t(src) * in_vec_size_g]);
      bi[row] = U(bi_p[size_t(src) * in_vec_size_g]);
    }
    _Pragma("clang loop unroll(full)")
    for (int b = 0; b < RB; b++) {
      if (FULL || b < nb) {
        uint4 xpk = ((const device uint4*)(xp0 + size_t(b) * K))[0];
        float4 xv0 = float4(as_type<half4>(xpk.xy));
        float4 xv1 = float4(as_type<half4>(xpk.zw));
        U smb = (xv0.x + xv0.y + xv0.z + xv0.w) + (xv1.x + xv1.y + xv1.z + xv1.w);
        _Pragma("clang loop unroll(full)")
        for (int row = 0; row < RPS; row++) {
          U accum = dot(xv0, w0[row]) + dot(xv1, w1[row]);
          result[b][row] += sc[row] * accum + smb * bi[row];
        }
      }
    }
    ws += 128;
    sc_p += 4;
    bi_p += 4;
    xp0 += block_size;
  }
""" + epi

_MARKOV_BEAM2_P2_SRC = r"""
  uint lid = thread_position_in_threadgroup.x;
  uint sg = lid >> 5;
  uint lane = lid & 31;
  uint m = sg >> 2;
  uint lsg = sg & 3;
  uint local_lid = lsg * 32 + lane;

  float fm = -INFINITY, fs = 0.0f;
  float bv[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
  uint bi4[4] = {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu};

  for (int i = int(local_lid); i < SP; i += 128) {
    size_t off = size_t(m) * SP + i;
    float om = pmax[off];
    if (om > -INFINITY) {
      float os = psum[off];
      float nm = max(fm, om);
      fs = (fm > -INFINITY ? fs * fast::exp(fm - nm) : 0.0f) + os * fast::exp(om - nm);
      fm = nm;
    }
    for (int k = 0; k < 4; k++) {
      float v = pval[off * 4 + k];
      uint idx = pidx[off * 4 + k];
      if (v > bv[3] || (v == bv[3] && idx < bi4[3])) {
        int pos = 3;
        while (pos > 0 && (v > bv[pos - 1] || (v == bv[pos - 1] && idx < bi4[pos - 1]))) {
          bv[pos] = bv[pos - 1]; bi4[pos] = bi4[pos - 1]; pos--;
        }
        bv[pos] = v; bi4[pos] = idx;
      }
    }
  }

  for (uint d = 16; d > 0; d >>= 1) {
    float om = simd_shuffle_down(fm, d);
    float os = simd_shuffle_down(fs, d);
    float ovs[4] = {
      simd_shuffle_down(bv[0], d), simd_shuffle_down(bv[1], d),
      simd_shuffle_down(bv[2], d), simd_shuffle_down(bv[3], d),
    };
    uint ois[4] = {
      simd_shuffle_down(bi4[0], d), simd_shuffle_down(bi4[1], d),
      simd_shuffle_down(bi4[2], d), simd_shuffle_down(bi4[3], d),
    };
    if (om > -INFINITY) {
      float nm = max(fm, om);
      fs = (fm > -INFINITY ? fs * fast::exp(fm - nm) : 0.0f) + os * fast::exp(om - nm);
      fm = nm;
    }
    for (int k = 0; k < 4; k++) {
      float ov = ovs[k];
      uint oi = ois[k];
      if (ov > bv[3] || (ov == bv[3] && oi < bi4[3])) {
        int pos = 3;
        while (pos > 0 && (ov > bv[pos - 1] || (ov == bv[pos - 1] && oi < bi4[pos - 1]))) {
          bv[pos] = bv[pos - 1]; bi4[pos] = bi4[pos - 1]; pos--;
        }
        bv[pos] = ov; bi4[pos] = oi;
      }
    }
  }

  threadgroup float sm[8], ss[8], sv[32];
  threadgroup uint si[32];
  threadgroup uint tg_src[2];
  if (lane == 0) {
    uint slot = m * 4 + lsg;
    sm[slot] = fm;
    ss[slot] = fs;
    for (int k = 0; k < 4; k++) { sv[slot * 4 + k] = bv[k]; si[slot * 4 + k] = bi4[k]; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (lid == 0) {
    float all_jp[8];
    uint all_tok[8];
    for (uint bm = 0; bm < 2; bm++) {
      float gm = -INFINITY, gs = 0.0f;
      for (uint s = 0; s < 4; s++) {
        float om = sm[bm * 4 + s];
        if (om > -INFINITY) {
          float nm = max(gm, om);
          gs = (gm > -INFINITY ? gs * fast::exp(gm - nm) : 0.0f) + ss[bm * 4 + s] * fast::exp(om - nm);
          gm = nm;
        }
      }
      float lse = gm + metal::log(gs);
      float fbv[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
      uint fbi[4] = {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu};
      for (uint j = 0; j < 16; j++) {
        float v = sv[bm * 16 + j];
        uint idx = si[bm * 16 + j];
        if (v > fbv[3] || (v == fbv[3] && idx < fbi[3])) {
          int pos = 3;
          while (pos > 0 && (v > fbv[pos - 1] || (v == fbv[pos - 1] && idx < fbi[pos - 1]))) {
            fbv[pos] = fbv[pos - 1]; fbi[pos] = fbi[pos - 1]; pos--;
          }
          fbv[pos] = v; fbi[pos] = idx;
        }
      }
      float jp_p = jp_prev[bm];
      int par_p = pidx_prev[bm];
      for (uint k = 0; k < 4; k++) {
        float prob = (fbv[k] > -INFINITY) ? fast::exp(fbv[k] - lse) : 0.0f;
        float jp_c = jp_p * prob;
        uint gi = bm * 4 + k;
        all_jp[gi] = jp_c;
        all_tok[gi] = fbi[k];
        ctok[gi] = fbi[k];
        cjp[gi] = jp_c;
        cpar[gi] = par_p;
      }
    }
    float v0 = -1.0f, v1 = -1.0f;
    uint b0 = 0, b1 = 1;
    for (uint i = 0; i < 8; i++) {
      float v = all_jp[i];
      if (v > v0) {
        v1 = v0; b1 = b0;
        v0 = v; b0 = i;
      } else if (v > v1) {
        v1 = v; b1 = i;
      }
    }
    next_tok[0] = all_tok[b0];
    next_tok[1] = all_tok[b1];
    next_jp[0] = v0;
    next_jp[1] = v1;
    next_pidx[0] = POOL_BASE + int(b0);
    next_pidx[1] = POOL_BASE + int(b1);
    tg_src[0] = b0 >> 2;
    tg_src[1] = b1 >> 2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint s0 = tg_src[0], s1 = tg_src[1];
  for (int i = int(lid); i < RR; i += 256) {
    next_st[i] = st_d[s0 * RR + i];
    next_st[RR + i] = st_d[s1 * RR + i];
  }
"""

_QMV4_FP16_RO_SRC = r"""
  constexpr int group_size = 64;
  constexpr int block_size = 256;
  static_assert(K % block_size == 0, "qmv4 fp16 ro needs K % 256 == 0");
  constexpr bool FULL = (B % RB) == 0;

  uint ty = threadgroup_position_in_grid.y;
  uint b0 = threadgroup_position_in_grid.z * RB;
  uint simd_gid = simdgroup_index_in_threadgroup;
  uint simd_lid = thread_index_in_simdgroup;

  const int out_row = int(ty) * (NSG * RPS) + int(simd_gid) * RPS;
  int last_row = RPS - 1;
  if (PR) {
    if (out_row >= N) return;
    last_row = min(last_row, N - 1 - out_row);
  }

  const int nb = FULL ? RB : min(RB, B - int(b0));
  thread float result[RB][RPS];
  for (int b = 0; b < RB; b++)
    for (int r = 0; r < RPS; r++) result[b][r] = 0.0f;

  const int in_vec_size_w = K / 2;
  const int in_vec_size_g = K / group_size;

  const device uint8_t* ws = (const device uint8_t*)w + size_t(out_row) * in_vec_size_w + simd_lid * 4;
  const device half* sc_p = scales + size_t(out_row) * in_vec_size_g + (simd_lid >> 3);
  const device half* bi_p = biases + size_t(out_row) * in_vec_size_g + (simd_lid >> 3);
  const device half* xp0 = x + size_t(b0) * K + simd_lid * 8;
  device half* yp = y + size_t(b0) * N + out_row;

  for (int k = 0; k < K; k += block_size) {
    _Pragma("clang loop unroll(full)")
    for (int row = 0; row < RPS; row++) {
      int src = PR ? min(row, last_row) : row;
      uint32_t wv = ((const device uint32_t*)(ws + size_t(src) * in_vec_size_w))[0];
      float sc = float(sc_p[size_t(src) * in_vec_size_g]);
      float bi = float(bi_p[size_t(src) * in_vec_size_g]);
      float4 w0 = float4(float(wv & 0xFu), float((wv >> 4) & 0xFu), float((wv >> 8) & 0xFu), float((wv >> 12) & 0xFu)) * sc + bi;
      float4 w1 = float4(float((wv >> 16) & 0xFu), float((wv >> 20) & 0xFu), float((wv >> 24) & 0xFu), float((wv >> 28) & 0xFu)) * sc + bi;
      _Pragma("clang loop unroll(full)")
      for (int b = 0; b < RB; b++) {
        if (FULL || b < nb) {
          uint4 xpk = ((const device uint4*)(xp0 + size_t(b) * K))[0];
          float4 xv0 = float4(as_type<half4>(xpk.xy));
          float4 xv1 = float4(as_type<half4>(xpk.zw));
          result[b][row] += dot(xv0, w0) + dot(xv1, w1);
        }
      }
    }
    ws += 128;
    sc_p += 4;
    bi_p += 4;
    xp0 += block_size;
  }

  _Pragma("clang loop unroll(full)")
  for (int b = 0; b < RB; b++) {
    _Pragma("clang loop unroll(full)")
    for (int row = 0; row < RPS; row++) {
      float r = simd_sum(result[b][row]);
      if (simd_lid == 0 && row <= last_row && (FULL || b < nb)) {
        yp[size_t(b) * N + row] = static_cast<half>(r);
      }
    }
  }
"""


def _qkv_norm_rope_f16_src(theta_lit):
    return f"""
  uint bh = threadgroup_position_in_grid.x;
  uint lane = thread_index_in_simdgroup;
  const int tot_h = NH + 2 * NKV;
  const int b = int(bh) / tot_h;
  const int h = int(bh) % tot_h;
  const device half* xp = qkv + size_t(bh) * 128;
  const int i0 = int(lane) * 2;
  const int j0 = i0 + 64;

  half2 xlo_h = *(const device half2*)(xp + i0);
  half2 xhi_h = *(const device half2*)(xp + j0);

  if (h >= NH + NKV) {{
    int vh = h - (NH + NKV);
    device half* vp = v + (size_t(vh) * BB + b) * 128;
    *(device half2*)(vp + i0) = xlo_h;
    *(device half2*)(vp + j0) = xhi_h;
    return;
  }}

  float2 flo = float2(xlo_h);
  float2 fhi = float2(xhi_h);
  float ss = simd_sum(flo.x * flo.x + flo.y * flo.y + fhi.x * fhi.x + fhi.y * fhi.y);
  float inv_rms = metal::rsqrt(ss / 128.0f + 1e-6f);
  const device half* nw = (h < NH) ? q_norm : k_norm;
  float2 wlo = float2(*(const device half2*)(nw + i0));
  float2 whi = float2(*(const device half2*)(nw + j0));
  float2 rlo = float2(half2(half(flo.x * inv_rms) * half(wlo.x), half(flo.y * inv_rms) * half(wlo.y)));
  float2 rhi = float2(half2(half(fhi.x * inv_rms) * half(whi.x), half(fhi.y * inv_rms) * half(whi.y)));

  const float p = float(pos_arr[0] + b);
  const float inv_half_d = 1.0f / 64.0f;
  const float theta = {theta_lit};
  float a0 = p * metal::pow(theta, -float(i0) * inv_half_d);
  float a1 = p * metal::pow(theta, -float(i0 + 1) * inv_half_d);
  float c0 = metal::cos(a0), s0 = metal::sin(a0);
  float c1 = metal::cos(a1), s1 = metal::sin(a1);

  half2 olo = half2(half(rlo.x * c0 - rhi.x * s0), half(rlo.y * c1 - rhi.y * s1));
  half2 ohi = half2(half(rhi.x * c0 + rlo.x * s0), half(rhi.y * c1 + rlo.y * s1));
  device half* dst = (h < NH) ? (q + (size_t(h) * BB + b) * 128) : (k + (size_t(h - NH) * BB + b) * 128);
  *(device half2*)(dst + i0) = olo;
  *(device half2*)(dst + j0) = ohi;
"""


_qmv4_k = None
_qmv4_ro_k = None
_qmv4_fp16_k = {}
_qmv4_fp16_ro_k = None
_qkv_norm_rope_f16_k = {}
_sdpa_draft_k = None
_sdpa_draft_qkv_k = {}
_qmv4_fp16_topk2_k = {}
_qmv4_bf16_ro_k = {}
_qmv4_bf16_u2_k = {}
_qmv4_bf16_mma8_k = {}
_qmv4_bf16_top16_p1_k = {}
_qmv4_bf16_top16_mma_p1_k = {}
_qmv4_bf16_top16_p2_k = None
_zero_pad_bf16_cache = {}
_dflash2_pack_chain_k = None
_dflash2_pack_tree_dink_k = None
_dyn_conv_s0_bf16_k = None
_dyn_conv_res_rms_bf16_k = None
_qkv_norm_rope_bf16_k = {}
_sdpa_draft_bf16_k = None
_sdpa_draft_bf16_q1_k = None
_topk2_chunks_k = None
_mk_sl_p1_k = {}
_draft_mk_pack_k = {}
_pack_tree_multi_dink_k: dict = {}
_mk_k = None
_mk_topk_k = {}
_mk_beam2_p2_k = None
_tree_walk_k = {}
_SCALE_F32_CACHE = {}


def _use_ro(B, N):
    ov = os.environ.get("LOOPLING_QMV4_RO")
    if ov is not None:
        return ov not in ("0", "") and B >= 2
    return N >= 32768 and B >= 2


def _cfg(B, N, ro=False):
    ov = os.environ.get("LOOPLING_QMV4_CFG")
    if ov:
        rps, nsg, rb = (int(t) for t in ov.split(","))
        return rps, nsg, min(rb, B)
    if B == 1:
        return 1, 2, 1
    if ro:
        return 4, 4, min(8, B)
    if N >= 16384:
        return 4, 4, min(8, B)
    return 4, 2, min(8, B)


def _get_qmv4(ro=False):
    global _qmv4_k, _qmv4_ro_k
    if ro:
        if _qmv4_ro_k is None:
            _qmv4_ro_k = mx.fast.metal_kernel(
                name="rows_qmv4_aff4_gs64_ro",
                input_names=["x", "w", "scales", "biases"],
                output_names=["y"],
                source=_QMV4_RO_SRC,
                ensure_row_contiguous=True,
            )
        return _qmv4_ro_k
    if _qmv4_k is None:
        _qmv4_k = mx.fast.metal_kernel(
            name="rows_qmv4_aff4_gs64",
            input_names=["x", "w", "scales", "biases"],
            output_names=["y"],
            source=_QMV4_SRC,
            ensure_row_contiguous=True,
        )
    return _qmv4_k


def qmv4_rows(x, w, scales, biases):
    """(B, K) bf16 x (N, K//8) u32 4-bit affine gs=64 -> (B, N) bf16, weights streamed once across up to 8 rows."""
    if x.ndim != 2 or x.dtype != mx.bfloat16:
        raise ValueError(f"qmv4_rows: x must be (B, K) bf16, got {x.shape} {x.dtype}")
    B, K = x.shape
    N = w.shape[0]
    if not 1 <= B <= MAX_B or K % 256 != 0:
        raise ValueError(f"qmv4_rows: unsupported B={B}, K={K}")
    ro = _use_ro(B, N)
    rps, nsg, rb = _cfg(B, N, ro=ro)
    tiles = (N + rps * nsg - 1) // (rps * nsg)
    pr = (N % (rps * nsg)) != 0
    return _get_qmv4(ro=ro)(
        inputs=[x, w, scales, biases],
        template=[("B", B), ("K", K), ("N", N), ("PR", pr), ("RPS", rps), ("NSG", nsg), ("RB", rb)],
        grid=(32, nsg * tiles, (B + rb - 1) // rb),
        threadgroup=(32, nsg, 1),
        output_shapes=[(B, N)],
        output_dtypes=[mx.bfloat16],
    )[0]


def _get_qmv4_fp16(swiglu=False):
    k = _qmv4_fp16_k.get(swiglu)
    if k is None:
        k = _qmv4_fp16_k[swiglu] = mx.fast.metal_kernel(
            name="rows_qmv4_aff4_gs64_fp16_sw" if swiglu else "rows_qmv4_aff4_gs64_fp16_bo",
            input_names=["x", "w", "scales", "biases"],
            output_names=["y"],
            source=_qmv4_fp16_src(swiglu=swiglu),
            ensure_row_contiguous=True,
        )
    return k


def _get_qmv4_fp16_ro():
    global _qmv4_fp16_ro_k
    if _qmv4_fp16_ro_k is None:
        _qmv4_fp16_ro_k = mx.fast.metal_kernel(
            name="rows_qmv4_aff4_gs64_fp16_ro",
            input_names=["x", "w", "scales", "biases"],
            output_names=["y"],
            source=_QMV4_FP16_RO_SRC,
            ensure_row_contiguous=True,
        )
    return _qmv4_fp16_ro_k


def _run_qmv4_fp16(x, w, scales, biases, swiglu=False):
    if x.ndim != 2 or x.dtype != mx.float16:
        raise ValueError(f"qmv4_fp16_rows: x must be (B, K) float16, got {x.shape} {x.dtype}")
    if scales.dtype != mx.float16 or biases.dtype != mx.float16:
        raise ValueError(f"qmv4_fp16_rows: scales/biases must be float16, got {scales.dtype}/{biases.dtype}")
    B, K = x.shape
    N = w.shape[0]
    if not 1 <= B <= MAX_B or K % 256 != 0:
        raise ValueError(f"qmv4_fp16_rows: unsupported B={B}, K={K}")
    if swiglu and N % 2 != 0:
        raise ValueError(f"qmv4_fp16_swiglu_rows: N must be even, got {N}")
    use_ro = (
        not swiglu
        and B >= 3
        and N >= 32768
        and os.environ.get("LOOPLING_QMV4_FP16_RO", "1") not in ("0", "")
    )
    ov = os.environ.get("LOOPLING_QMV4_FP16_CFG")
    if ov:
        rps, nsg, rb = (int(t) for t in ov.split(","))
        rb = min(rb, B)
    elif B == 1:
        rps, nsg, rb = (2, 2, 1) if swiglu else (1, 2, 1)
    elif use_ro:
        rps, nsg, rb = 4, 4, min(8, B)
    else:
        rps, nsg, rb = 4, 4, min(8, B)
    if swiglu and rps % 2 != 0:
        raise ValueError(f"qmv4_fp16_swiglu_rows: RPS must be even, got {rps}")
    tiles = (N + rps * nsg - 1) // (rps * nsg)
    pr = (N % (rps * nsg)) != 0
    out_n = (N // 2) if swiglu else N
    kfn = _get_qmv4_fp16_ro() if use_ro else _get_qmv4_fp16(swiglu=swiglu)
    return kfn(
        inputs=[x, w, scales, biases],
        template=[("B", B), ("K", K), ("N", N), ("PR", pr), ("RPS", rps), ("NSG", nsg), ("RB", rb)],
        grid=(32, nsg * tiles, (B + rb - 1) // rb),
        threadgroup=(32, nsg, 1),
        output_shapes=[(B, out_n)],
        output_dtypes=[mx.float16],
    )[0]


def qkv_norm_rope_f16(qkv, q_norm, k_norm, pos_arr, nh, nkv, hd, theta):
    """Fused fp16 Q/K RMSNorm + NeoX RoPE + V transpose for (B, (NH+2*NKV)*hd) -> q, k, v."""
    assert hd == 128 and qkv.ndim == 2 and qkv.dtype == mx.float16
    BB = int(qkv.shape[0])
    NH, NKV = int(nh), int(nkv)
    assert qkv.shape[1] == (NH + 2 * NKV) * 128
    th_f = float(theta)
    kfn = _qkv_norm_rope_f16_k.get(th_f)
    if kfn is None:
        th_lit = flit(th_f)
        sfx = th_lit.replace(".", "p").replace("-", "m").replace("+", "P")
        kfn = _qkv_norm_rope_f16_k[th_f] = mx.fast.metal_kernel(
            name=f"qkv_norm_rope_f16_{sfx}",
            input_names=["qkv", "q_norm", "k_norm", "pos_arr"],
            output_names=["q", "k", "v"],
            source=_qkv_norm_rope_f16_src(th_lit),
            ensure_row_contiguous=True,
        )
    return kfn(
        inputs=[qkv, q_norm, k_norm, pos_arr.reshape(-1)],
        template=[("BB", BB), ("NH", NH), ("NKV", NKV)],
        grid=(BB * (NH + 2 * NKV) * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(1, NH, BB, 128), (1, NKV, BB, 128), (1, NKV, BB, 128)],
        output_dtypes=[mx.float16, mx.float16, mx.float16],
    )


def qmv4_fp16_rows(x, w, scales, biases):
    """(B, K) fp16 x (N, K//8) u32 4-bit affine gs=64 fp16 scales/biases -> (B, N) fp16."""
    return _run_qmv4_fp16(x, w, scales, biases, swiglu=False)


def qmv4_fp16_swiglu_rows(x, w, scales, biases):
    """(B, K) fp16 x interleaved (2M, K//8) u32 gate/up -> (B, M) fp16 SwiGLU."""
    return _run_qmv4_fp16(x, w, scales, biases, swiglu=True)


_SDPA_DRAFT_SRC = r"""
  uint bh = threadgroup_position_in_grid.x;
  uint sg = simdgroup_index_in_threadgroup;
  uint lane = thread_index_in_simdgroup;
  int b = int(bh) / HQ;
  int h = int(bh) % HQ;
  int kvh = h / GQA;

  const int P = dr_pos[0];
  const int s = (WIN > 0) ? max(0, P - WIN) : 0;
  const int n_ctx = max(0, P - s);
  const int tot = n_ctx + BB;

  float4 qv = float4(*(const device half4*)(q + (size_t(h) * BB + b) * D + lane * 4)) * scale[0];
  const device half* ck_h = ck + size_t(kvh) * CAP * D + lane * 4;
  const device half* cv_h = cv + size_t(kvh) * CAP * D + lane * 4;
  const device half* kc_h = k_cur + size_t(kvh) * BB * D + lane * 4;
  const device half* vc_h = v_cur + size_t(kvh) * BB * D + lane * 4;

  float m_sg = -INFINITY;
  float l_sg = 0.0f;
  float4 ov4 = float4(0.0f);

  for (int idx = int(sg); idx < tot; idx += NS) {
    const device half* kp = (idx < n_ctx) ? (ck_h + size_t(s + idx) * D) : (kc_h + size_t(idx - n_ctx) * D);
    const device half* vp = (idx < n_ctx) ? (cv_h + size_t(s + idx) * D) : (vc_h + size_t(idx - n_ctx) * D);
    float4 kv4 = float4(*(const device half4*)kp);
    float4 vv4 = float4(*(const device half4*)vp);
    float score = simd_sum(qv.x * kv4.x + qv.y * kv4.y + qv.z * kv4.z + qv.w * kv4.w);
    float new_max = max(m_sg, score);
    float factor = (m_sg > -INFINITY) ? fast::exp(m_sg - new_max) : 0.0f;
    float exp_s = fast::exp(score - new_max);
    m_sg = new_max;
    l_sg = l_sg * factor + exp_s;
    ov4 = ov4 * factor + exp_s * vv4;
  }

  threadgroup float tg_m[NS];
  threadgroup float tg_l[NS];
  threadgroup float4 tg_o[NS][32];
  if (lane == 0) {
    tg_m[sg] = m_sg;
    tg_l[sg] = l_sg;
  }
  tg_o[sg][lane] = ov4;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sg == 0) {
    float gm = -INFINITY;
    for (int s0 = 0; s0 < NS; s0++) gm = max(gm, tg_m[s0]);
    float gl = 0.0f;
    float4 go = float4(0.0f);
    for (int s0 = 0; s0 < NS; s0++) {
      float ms0 = tg_m[s0];
      if (ms0 > -INFINITY) {
        float w = fast::exp(ms0 - gm);
        gl += w * tg_l[s0];
        go += w * tg_o[s0][lane];
      }
    }
    go *= (gl > 0.0f) ? (1.0f / gl) : 0.0f;
    *(device half4*)(out + (size_t(b) * HQ + h) * D + lane * 4) = half4(go);
  }
"""


def sdpa_draft_gpu(q, ck, cv, k_cur, v_cur, dr_pos_arr, *, scale, window):
    """Exact fp16 bidirectional block attention over FastDrafter KV + current slots -> (B, HQ*D) fp16."""
    global _sdpa_draft_k
    if _sdpa_draft_k is None:
        _sdpa_draft_k = mx.fast.metal_kernel(
            name="sdpa_draft_fp16_gpu",
            input_names=["q", "ck", "cv", "k_cur", "v_cur", "dr_pos", "scale"],
            output_names=["out"],
            source=_SDPA_DRAFT_SRC,
            ensure_row_contiguous=True,
        )
    _, HQ, BB, D = q.shape
    HKV = int(k_cur.shape[1])
    CAP = int(ck.shape[2])
    assert D == 128 and HQ % HKV == 0
    sck = float(scale)
    sc = _SCALE_F32_CACHE.get(sck)
    if sc is None:
        sc = _SCALE_F32_CACHE[sck] = mx.array([sck], mx.float32)
    ns = 8
    return _sdpa_draft_k(
        inputs=[q, ck, cv, k_cur, v_cur, dr_pos_arr, sc],
        template=[("BB", BB), ("HQ", HQ), ("GQA", HQ // HKV), ("D", D), ("CAP", CAP), ("WIN", int(window)), ("NS", ns)],
        grid=(BB * HQ * 32 * ns, 1, 1),
        threadgroup=(32 * ns, 1, 1),
        output_shapes=[(BB, HQ * D)],
        output_dtypes=[mx.float16],
    )[0]


def _sdpa_draft_qkv_src(theta_lit):
    return f"""
  uint bh = threadgroup_position_in_grid.x;
  uint sg = simdgroup_index_in_threadgroup;
  uint lane = thread_index_in_simdgroup;
  int b = int(bh) / HQ;
  int h = int(bh) % HQ;
  int kvh = h / GQA;

  const int q_dim = HQ * 128;
  const int kv_dim = HKV * 128;
  const int qkv_stride = q_dim + 2 * kv_dim;
  const int i0 = int(lane) * 2;
  const int j0 = i0 + 64;

  const int P = dr_pos[0];
  const int s = (WIN > 0) ? max(0, P - WIN) : 0;
  const int n_ctx = max(0, P - s);
  const int tot = n_ctx + BB;

  const float inv_half_d = 1.0f / 64.0f;
  const float theta = {theta_lit};
  const float freq0 = metal::pow(theta, -float(i0) * inv_half_d);
  const float freq1 = metal::pow(theta, -float(i0 + 1) * inv_half_d);

  // Q RMSNorm + half-split RoPE for slot b, head h
  const device half* qp = qkv + size_t(b) * qkv_stride + size_t(h) * 128;
  float2 q_flo = float2(*(const device half2*)(qp + i0));
  float2 q_fhi = float2(*(const device half2*)(qp + j0));
  float q_ss = simd_sum(q_flo.x * q_flo.x + q_flo.y * q_flo.y + q_fhi.x * q_fhi.x + q_fhi.y * q_fhi.y);
  float q_inv = metal::rsqrt(q_ss / 128.0f + 1e-6f);
  float2 qw_lo = float2(*(const device half2*)(q_norm + i0));
  float2 qw_hi = float2(*(const device half2*)(q_norm + j0));
  float2 q_rlo = float2(half2(half(q_flo.x * q_inv) * half(qw_lo.x), half(q_flo.y * q_inv) * half(qw_lo.y)));
  float2 q_rhi = float2(half2(half(q_fhi.x * q_inv) * half(qw_hi.x), half(q_fhi.y * q_inv) * half(qw_hi.y)));
  float pb_pos = float(P + b);
  float qa0 = pb_pos * freq0, qa1 = pb_pos * freq1;
  float qc0 = metal::cos(qa0), qs0 = metal::sin(qa0);
  float qc1 = metal::cos(qa1), qs1 = metal::sin(qa1);
  float2 qflo = float2(half2(half(q_rlo.x * qc0 - q_rhi.x * qs0), half(q_rlo.y * qc1 - q_rhi.y * qs1))) * scale[0];
  float2 qfhi = float2(half2(half(q_rhi.x * qc0 + q_rlo.x * qs0), half(q_rhi.y * qc1 + q_rlo.y * qs1))) * scale[0];

  const device half* ck_h = ck + size_t(kvh) * CAP * 128;
  const device half* cv_h = cv + size_t(kvh) * CAP * 128;
  float2 kw_lo = float2(*(const device half2*)(k_norm + i0));
  float2 kw_hi = float2(*(const device half2*)(k_norm + j0));

  float m_sg = -INFINITY;
  float l_sg = 0.0f;
  float2 ov_lo = float2(0.0f);
  float2 ov_hi = float2(0.0f);

  for (int idx = int(sg); idx < tot; idx += NS) {{
    float2 fklo, fkhi, fvlo, fvhi;
    if (idx < n_ctx) {{
      size_t off = size_t(s + idx) * 128;
      fklo = float2(*(const device half2*)(ck_h + off + i0));
      fkhi = float2(*(const device half2*)(ck_h + off + j0));
      fvlo = float2(*(const device half2*)(cv_h + off + i0));
      fvhi = float2(*(const device half2*)(cv_h + off + j0));
    }} else {{
      int bk = idx - n_ctx;
      const device half* kp = qkv + size_t(bk) * qkv_stride + q_dim + size_t(kvh) * 128;
      const device half* vp = kp + kv_dim;
      float2 k_flo = float2(*(const device half2*)(kp + i0));
      float2 k_fhi = float2(*(const device half2*)(kp + j0));
      float k_ss = simd_sum(k_flo.x * k_flo.x + k_flo.y * k_flo.y + k_fhi.x * k_fhi.x + k_fhi.y * k_fhi.y);
      float k_inv = metal::rsqrt(k_ss / 128.0f + 1e-6f);
      float2 k_rlo = float2(half2(half(k_flo.x * k_inv) * half(kw_lo.x), half(k_flo.y * k_inv) * half(kw_lo.y)));
      float2 k_rhi = float2(half2(half(k_fhi.x * k_inv) * half(kw_hi.x), half(k_fhi.y * k_inv) * half(kw_hi.y)));
      float kpos = float(P + bk);
      float ka0 = kpos * freq0, ka1 = kpos * freq1;
      float kc0 = metal::cos(ka0), ks0 = metal::sin(ka0);
      float kc1 = metal::cos(ka1), ks1 = metal::sin(ka1);
      fklo = float2(half2(half(k_rlo.x * kc0 - k_rhi.x * ks0), half(k_rlo.y * kc1 - k_rhi.y * ks1)));
      fkhi = float2(half2(half(k_rhi.x * kc0 + k_rlo.x * ks0), half(k_rhi.y * kc1 + k_rlo.y * ks1)));
      fvlo = float2(*(const device half2*)(vp + i0));
      fvhi = float2(*(const device half2*)(vp + j0));
    }}
    float score = simd_sum(qflo.x * fklo.x + qflo.y * fklo.y + qfhi.x * fkhi.x + qfhi.y * fkhi.y);
    float new_max = max(m_sg, score);
    float factor = (m_sg > -INFINITY) ? fast::exp(m_sg - new_max) : 0.0f;
    float exp_s = fast::exp(score - new_max);
    m_sg = new_max;
    l_sg = l_sg * factor + exp_s;
    ov_lo = ov_lo * factor + exp_s * fvlo;
    ov_hi = ov_hi * factor + exp_s * fvhi;
  }}

  threadgroup float tg_m[NS];
  threadgroup float tg_l[NS];
  threadgroup float2 tg_olo[NS][32];
  threadgroup float2 tg_ohi[NS][32];
  if (lane == 0) {{
    tg_m[sg] = m_sg;
    tg_l[sg] = l_sg;
  }}
  tg_olo[sg][lane] = ov_lo;
  tg_ohi[sg][lane] = ov_hi;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sg == 0) {{
    float gm = -INFINITY;
    for (int s0 = 0; s0 < NS; s0++) gm = max(gm, tg_m[s0]);
    float gl = 0.0f;
    float2 glo = float2(0.0f);
    float2 ghi = float2(0.0f);
    for (int s0 = 0; s0 < NS; s0++) {{
      float ms0 = tg_m[s0];
      if (ms0 > -INFINITY) {{
        float w = fast::exp(ms0 - gm);
        gl += w * tg_l[s0];
        glo += w * tg_olo[s0][lane];
        ghi += w * tg_ohi[s0][lane];
      }}
    }}
    float inv_gl = (gl > 0.0f) ? (1.0f / gl) : 0.0f;
    device half* dp = out + (size_t(b) * HQ + h) * 128;
    *(device half2*)(dp + i0) = half2(glo * inv_gl);
    *(device half2*)(dp + j0) = half2(ghi * inv_gl);
  }}
"""


def sdpa_draft_qkv_gpu(qkv, q_norm, k_norm, ck, cv, dr_pos_arr, *, nh, nkv, hd, theta, scale, window):
    """Fused Q/K RMSNorm + RoPE + bidirectional block SDPA directly from packed qkv -> (B, NH*D) fp16."""
    assert hd == 128 and nh % nkv == 0
    th_f = float(theta)
    k = _sdpa_draft_qkv_k.get(th_f)
    if k is None:
        th_lit = flit(th_f)
        k = _sdpa_draft_qkv_k[th_f] = mx.fast.metal_kernel(
            name="sdpa_draft_qkv_" + th_lit.replace(".", "p").replace("-", "m").replace("+", "P"),
            input_names=["qkv", "q_norm", "k_norm", "ck", "cv", "dr_pos", "scale"],
            output_names=["out"],
            source=_sdpa_draft_qkv_src(th_lit),
            ensure_row_contiguous=True,
        )
    BB = int(qkv.shape[0])
    CAP = int(ck.shape[2])
    sck = float(scale)
    sc = _SCALE_F32_CACHE.get(sck)
    if sc is None:
        sc = _SCALE_F32_CACHE[sck] = mx.array([sck], mx.float32)
    ns = 8
    return k(
        inputs=[qkv, q_norm, k_norm, ck, cv, dr_pos_arr.reshape(-1), sc],
        template=[("BB", BB), ("HQ", int(nh)), ("HKV", int(nkv)), ("GQA", int(nh) // int(nkv)), ("CAP", CAP), ("WIN", int(window)), ("NS", ns)],
        grid=(BB * int(nh) * 32 * ns, 1, 1),
        threadgroup=(32 * ns, 1, 1),
        output_shapes=[(BB, int(nh) * 128)],
        output_dtypes=[mx.float16],
    )[0]


_QMV4_FP16_TOPK2_SRC = r"""
  constexpr int group_size = 64;
  constexpr int block_size = 256;
  constexpr int NSG = 4;
  constexpr int RPS = 4;
  static_assert(K % block_size == 0, "qmv4 fp16 topk2 needs K % 256 == 0");

  uint sp_id = threadgroup_position_in_grid.y;
  uint simd_gid = simdgroup_index_in_threadgroup;
  uint simd_lid = thread_index_in_simdgroup;

  float bv0[BB], bv1[BB];
  uint  bi0[BB], bi1[BB];
  for (int b = 0; b < BB; b++) {
    bv0[b] = -INFINITY; bv1[b] = -INFINITY;
    bi0[b] = 0xFFFFFFFFu; bi1[b] = 0xFFFFFFFFu;
  }

  const int in_vec_size_w = K / 2;
  const int in_vec_size_g = K / group_size;

  for (int ty = int(sp_id); ty < TOT_TILES; ty += SP) {
    const int out_row = ty * 16 + int(simd_gid) * RPS;
    if (out_row < N) {
      const int last_row = min(RPS - 1, N - 1 - out_row);
      float result[BB][RPS];
      for (int b = 0; b < BB; b++)
        for (int r = 0; r < RPS; r++) result[b][r] = 0.0f;

      const device uint8_t* ws = (const device uint8_t*)w + size_t(out_row) * in_vec_size_w + simd_lid * 4;
      const device half* sc_p = scales + size_t(out_row) * in_vec_size_g + (simd_lid >> 3);
      const device half* bi_p = biases + size_t(out_row) * in_vec_size_g + (simd_lid >> 3);
      const device half* xp0 = x + simd_lid * 8;

      for (int k = 0; k < K; k += block_size) {
        float4 xv0[BB], xv1[BB];
        for (int b = 0; b < BB; b++) {
          uint4 xpk = ((const device uint4*)(xp0 + size_t(b) * K))[0];
          xv0[b] = float4(as_type<half4>(xpk.xy));
          xv1[b] = float4(as_type<half4>(xpk.zw));
        }
        _Pragma("clang loop unroll(full)")
        for (int row = 0; row < RPS; row++) {
          int src = min(row, last_row);
          uint32_t wv = ((const device uint32_t*)(ws + size_t(src) * in_vec_size_w))[0];
          float sc = float(sc_p[size_t(src) * in_vec_size_g]);
          float bi = float(bi_p[size_t(src) * in_vec_size_g]);
          float4 w0 = float4(float(wv & 0xFu), float((wv >> 4) & 0xFu), float((wv >> 8) & 0xFu), float((wv >> 12) & 0xFu)) * sc + bi;
          float4 w1 = float4(float((wv >> 16) & 0xFu), float((wv >> 20) & 0xFu), float((wv >> 24) & 0xFu), float((wv >> 28) & 0xFu)) * sc + bi;
          for (int b = 0; b < BB; b++) {
            result[b][row] += dot(xv0[b], w0) + dot(xv1[b], w1);
          }
        }
        ws += 128;
        sc_p += 4;
        bi_p += 4;
        xp0 += block_size;
      }

      for (int row = 0; row <= last_row; row++) {
        uint idx = uint(out_row + row);
        float fb = float(fold_b[idx]);
        for (int b = 0; b < BB; b++) {
          float r = simd_sum(result[b][row]);
          if (simd_lid == 0) {
            float val = float(static_cast<half>(r)) + fb;
            if (val > bv0[b] || (val == bv0[b] && idx < bi0[b])) {
              bv1[b] = bv0[b]; bi1[b] = bi0[b]; bv0[b] = val; bi0[b] = idx;
            } else if (val > bv1[b] || (val == bv1[b] && idx < bi1[b])) {
              bv1[b] = val; bi1[b] = idx;
            }
          }
        }
      }
    }
  }

  threadgroup float tg_v0[NSG][BB], tg_v1[NSG][BB];
  threadgroup uint  tg_i0[NSG][BB], tg_i1[NSG][BB];
  if (simd_lid == 0) {
    for (int b = 0; b < BB; b++) {
      tg_v0[simd_gid][b] = bv0[b]; tg_i0[simd_gid][b] = bi0[b];
      tg_v1[simd_gid][b] = bv1[b]; tg_i1[simd_gid][b] = bi1[b];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_lid == 0) {
    for (int b = int(simd_gid); b < BB; b += NSG) {
      float fv0 = -INFINITY, fv1 = -INFINITY;
      uint  fi0 = 0xFFFFFFFFu, fi1 = 0xFFFFFFFFu;
      for (int s = 0; s < NSG; s++) {
        float ov0 = tg_v0[s][b]; uint oi0 = tg_i0[s][b];
        float ov1 = tg_v1[s][b]; uint oi1 = tg_i1[s][b];
        if (ov0 > fv0 || (ov0 == fv0 && oi0 < fi0)) {
          fv1 = fv0; fi1 = fi0; fv0 = ov0; fi0 = oi0;
          if (ov1 > fv1 || (ov1 == fv1 && oi1 < fi1)) { fv1 = ov1; fi1 = oi1; }
        } else if (ov0 > fv1 || (ov0 == fv1 && oi0 < fi1)) {
          fv1 = ov0; fi1 = oi0;
        }
      }
      size_t o = size_t(b) * (2 * SP) + sp_id * 2;
      top_val[o + 0] = fv0; top_idx[o + 0] = fi0;
      top_val[o + 1] = fv1; top_idx[o + 1] = fi1;
    }
  }
"""


def qmv4_fp16_topk2_rows(x, w, scales, biases, fold_b_f16, sp=256):
    """Fused 4-bit fp16 LM head + fold_b + per-bucket Top-2 -> (top_val (B, 2*sp) f32, top_idx (B, 2*sp) u32)."""
    BB, K = int(x.shape[0]), int(x.shape[1])
    N = int(w.shape[0])
    assert 1 <= BB <= 5 and K % 256 == 0 and x.dtype == mx.float16
    assert w.shape[1] * 8 == K and int(fold_b_f16.shape[0]) == N and fold_b_f16.dtype == mx.float16
    k = _qmv4_fp16_topk2_k.get(BB)
    if k is None:
        k = _qmv4_fp16_topk2_k[BB] = mx.fast.metal_kernel(
            name=f"qmv4_fp16_topk2_b{BB}",
            input_names=["x", "w", "scales", "biases", "fold_b"],
            output_names=["top_val", "top_idx"],
            source=_QMV4_FP16_TOPK2_SRC,
            ensure_row_contiguous=True,
        )
    tot_tiles = (N + 15) // 16
    k_out = 2 * int(sp)
    return k(
        inputs=[x, w, scales, biases, fold_b_f16],
        template=[("BB", BB), ("K", K), ("N", N), ("SP", int(sp)), ("TOT_TILES", tot_tiles)],
        grid=(32, 4 * int(sp), 1),
        threadgroup=(32, 4, 1),
        output_shapes=[(BB, k_out), (BB, k_out)],
        output_dtypes=[mx.float32, mx.uint32],
    )


_TOPK2_CHUNKS_SRC = r"""
  uint sp = threadgroup_position_in_grid.x;
  uint b = threadgroup_position_in_grid.y;
  uint lane = thread_index_in_simdgroup;

  float bv0 = -INFINITY, bv1 = -INFINITY;
  uint  bi0 = 0xFFFFFFFFu, bi1 = 0xFFFFFFFFu;
  const device half* brow = base + size_t(b) * VV;

  if (lane < 16) {
    for (int ty = int(sp); ty < TOT_TILES; ty += SP) {
      uint idx = uint(ty * 16 + int(lane));
      if (int(idx) < VV) {
        float val = float(brow[idx]) + float(fold_b[idx]);
        if (val > bv0 || (val == bv0 && idx < bi0)) {
          bv1 = bv0; bi1 = bi0; bv0 = val; bi0 = idx;
        } else if (val > bv1 || (val == bv1 && idx < bi1)) {
          bv1 = val; bi1 = idx;
        }
      }
    }
  }

  for (uint d = 8; d > 0; d >>= 1) {
    float ov0 = simd_shuffle_down(bv0, d); uint oi0 = simd_shuffle_down(bi0, d);
    float ov1 = simd_shuffle_down(bv1, d); uint oi1 = simd_shuffle_down(bi1, d);
    if (ov0 > bv0 || (ov0 == bv0 && oi0 < bi0)) {
      bv1 = bv0; bi1 = bi0; bv0 = ov0; bi0 = oi0;
      if (ov1 > bv1 || (ov1 == bv1 && oi1 < bi1)) { bv1 = ov1; bi1 = oi1; }
    } else if (ov0 > bv1 || (ov0 == bv1 && oi0 < bi1)) {
      bv1 = ov0; bi1 = oi0;
    }
  }

  if (lane == 0) {
    size_t o = size_t(b) * (2 * SP) + sp * 2;
    top_val[o + 0] = bv0; top_idx[o + 0] = bi0;
    top_val[o + 1] = bv1; top_idx[o + 1] = bi1;
  }
"""


def topk2_chunks_f16(base, fold_b_f16, sp=256):
    """Extract Top-2 (base + fold_b_f16) from each 16-row strided tile bucket -> (D, 2*sp) f32/u32."""
    global _topk2_chunks_k
    if _topk2_chunks_k is None:
        _topk2_chunks_k = mx.fast.metal_kernel(
            name="topk2_chunks_f16",
            input_names=["base", "fold_b"],
            output_names=["top_val", "top_idx"],
            source=_TOPK2_CHUNKS_SRC,
            ensure_row_contiguous=True,
        )
    BB, VV = int(base.shape[0]), int(base.shape[1])
    tot_tiles = (VV + 15) // 16
    k_out = 2 * int(sp)
    return _topk2_chunks_k(
        inputs=[base, fold_b_f16],
        template=[("VV", VV), ("SP", int(sp)), ("TOT_TILES", tot_tiles)],
        grid=(int(sp) * 32, BB, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(BB, k_out), (BB, k_out)],
        output_dtypes=[mx.float32, mx.uint32],
    )


def _markov_sl_p1_src(scap_lit, inv_t_lit):
    return f"""
  uint sp = threadgroup_position_in_grid.x;
  uint lid = thread_position_in_threadgroup.x;
  uint lane = lid & 31, sg = lid >> 5;

  float4 lv0[MM], lv1[MM];
  for (int m = 0; m < MM; m++) {{
    uint4 lpk = ((const device uint4*)(latent + size_t(m) * 256 + lane * 8))[0];
    lv0[m] = float4(as_type<bfloat4>(lpk.xy));
    lv1[m] = float4(as_type<bfloat4>(lpk.zw));
  }}

  const int chunk = (KK + SP - 1) / SP;
  const int r0 = int(sp) * chunk;
  const int r1 = min(KK, r0 + chunk);

  float tv[MM][4];
  uint ti[MM][4];
  float smax[MM], ssum[MM];
  for (int m = 0; m < MM; m++) {{
    for (int k = 0; k < 4; k++) {{ tv[m][k] = -INFINITY; ti[m][k] = 0xFFFFFFFFu; }}
    smax[m] = -INFINITY; ssum[m] = 0.0f;
  }}

  const float sc_cap = {scap_lit};
  const float inv_cap = 1.0f / {scap_lit};
  const float inv_t = {inv_t_lit};

  for (int i = r0 + int(sg); i < r1; i += NSG) {{
    uint tok_id = top_idx[i];
    float bf = top_val[i];
    uint4 wpk = ((const device uint4*)(fold_w + size_t(tok_id) * 256 + lane * 8))[0];
    float4 w0 = float4(as_type<half4>(wpk.xy));
    float4 w1 = float4(as_type<half4>(wpk.zw));
    for (int m = 0; m < MM; m++) {{
      float val = simd_sum(dot(lv0[m], w0) + dot(lv1[m], w1));
      if (lane == 0) {{
        float lg = sc_cap * fast::tanh((bf + val) * inv_cap) * inv_t;
        float nm = max(smax[m], lg);
        ssum[m] = (smax[m] > -INFINITY ? ssum[m] * fast::exp(smax[m] - nm) : 0.0f) + fast::exp(lg - nm);
        smax[m] = nm;
        if (lg > tv[m][3] || (lg == tv[m][3] && tok_id < ti[m][3])) {{
          int pos = 3;
          while (pos > 0 && (lg > tv[m][pos - 1] || (lg == tv[m][pos - 1] && tok_id < ti[m][pos - 1]))) {{
            tv[m][pos] = tv[m][pos - 1]; ti[m][pos] = ti[m][pos - 1]; pos--;
          }}
          tv[m][pos] = lg; ti[m][pos] = tok_id;
        }}
      }}
    }}
  }}

  threadgroup float tgv[NSG * 4];
  threadgroup uint tgi[NSG * 4];
  threadgroup float tgm[NSG];
  threadgroup float tgs[NSG];
  for (int m = 0; m < MM; m++) {{
    if (lane == 0) {{
      for (int k = 0; k < 4; k++) {{ tgv[sg * 4 + k] = tv[m][k]; tgi[sg * 4 + k] = ti[m][k]; }}
      tgm[sg] = smax[m]; tgs[sg] = ssum[m];
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane == 0) {{
      float fm = -INFINITY, fs = 0.0f;
      for (int s = 0; s < NSG; s++) {{
        float om = tgm[s];
        if (om > -INFINITY) {{
          float nm = max(fm, om);
          fs = (fm > -INFINITY ? fs * fast::exp(fm - nm) : 0.0f) + tgs[s] * fast::exp(om - nm);
          fm = nm;
        }}
      }}
      pmax[size_t(m) * SP + sp] = fm;
      psum[size_t(m) * SP + sp] = fs;
      float bv[4] = {{-INFINITY, -INFINITY, -INFINITY, -INFINITY}};
      uint bi4[4] = {{0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu}};
      for (int j = 0; j < NSG * 4; j++) {{
        float v = tgv[j]; uint idx = tgi[j];
        if (v > bv[3] || (v == bv[3] && idx < bi4[3])) {{
          int pos = 3;
          while (pos > 0 && (v > bv[pos - 1] || (v == bv[pos - 1] && idx < bi4[pos - 1]))) {{
            bv[pos] = bv[pos - 1]; bi4[pos] = bi4[pos - 1]; pos--;
          }}
          bv[pos] = v; bi4[pos] = idx;
        }}
      }}
      for (int k = 0; k < 4; k++) {{
        pval[(size_t(m) * SP + sp) * 4 + k] = bv[k];
        pidx[(size_t(m) * SP + sp) * 4 + k] = bi4[k];
      }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }}
"""


def _get_markov_sl_p1(softcap, inv_t):
    key = (float(softcap), float(inv_t))
    k = _mk_sl_p1_k.get(key)
    if k is None:
        sc_lit = flit(key[0])
        it_lit = flit(key[1])
        sfx = f"{sc_lit}_{it_lit}".replace(".", "p").replace("-", "m").replace("+", "P")
        k = _mk_sl_p1_k[key] = mx.fast.metal_kernel(
            name=f"markov_sl_p1_f16_{sfx}",
            input_names=["latent", "top_val", "top_idx", "fold_w"],
            output_names=["pval", "pidx", "pmax", "psum"],
            source=_markov_sl_p1_src(sc_lit, it_lit),
            ensure_row_contiguous=True,
        )
    return k


def markov_shortlist_topk4_f16(latent, top_val, top_idx, fold_w_f16, softcap, temp=1.6):
    """Shortlist top-4 tokens & softmax probs for M=1 latent over prefiltered K candidates."""
    kk = int(top_val.shape[0])
    sp = 16 if kk <= 256 else 32
    nsg = 4
    p1 = _get_markov_sl_p1(softcap, 1.0 / float(temp))
    _, p2 = _get_markov_topk(softcap)
    pval, pidx, pmax, psum = p1(
        inputs=[latent.reshape(-1), top_val.reshape(-1), top_idx.reshape(-1), fold_w_f16],
        template=[("KK", kk), ("SP", sp), ("NSG", nsg), ("MM", 1)],
        grid=(sp * 32 * nsg, 1, 1),
        threadgroup=(32 * nsg, 1, 1),
        output_shapes=[(1, sp, 4), (1, sp, 4), (1, sp), (1, sp)],
        output_dtypes=[mx.float32, mx.uint32, mx.float32, mx.float32],
    )
    return p2(
        inputs=[pval, pidx, pmax, psum],
        template=[("SP", sp)],
        grid=(128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(1, 4), (1, 4)],
        output_dtypes=[mx.uint32, mx.float32],
    )


def _get_markov():
    global _mk_k
    if _mk_k is None:
        p1 = mx.fast.metal_kernel(
            name="markov_bias_argmax_4b_p1",
            input_names=["latent", "fw_w", "fw_s", "fw_b", "fold_b", "base_row"],
            output_names=["pval", "pidx"],
            source=_MARKOV_P1_SRC,
            ensure_row_contiguous=True,
        )
        p2 = mx.fast.metal_kernel(
            name="markov_bias_argmax_4b_p2",
            input_names=["pval", "pidx"],
            output_names=["out"],
            source=_MARKOV_P2_SRC,
            ensure_row_contiguous=True,
        )
        _mk_k = (p1, p2)
    return _mk_k


def _get_markov_topk(softcap):
    sc_f = float(softcap)
    entry = _mk_topk_k.get(sc_f)
    if entry is None:
        scap_lit = flit(sc_f)
        sfx = scap_lit.replace(".", "p").replace("-", "m").replace("+", "P")
        p1 = mx.fast.metal_kernel(
            name=f"markov_bias_topk4_4b_p1_{sfx}",
            input_names=["latent", "fw_w", "fw_s", "fw_b", "fold_b", "base_row"],
            output_names=["pval", "pidx", "pmax", "psum"],
            source=_markov_topk4_p1_src(scap_lit),
            ensure_row_contiguous=True,
        )
        p2 = mx.fast.metal_kernel(
            name="markov_bias_topk4_4b_p2",
            input_names=["pval", "pidx", "pmax", "psum"],
            output_names=["out_tok", "out_prob"],
            source=_MARKOV_TOPK4_P2_SRC,
            ensure_row_contiguous=True,
        )
        entry = _mk_topk_k[sc_f] = (p1, p2)
    return entry


def markov_bias_argmax_4b(latent, fw_w, fw_s, fw_b, fold_b, base_row, vocab):
    """argmax_{r < vocab}(base_row[r] + fold_b[r] + latent @ fold_w_4b[r]) returning (1,) uint32."""
    p1, p2 = _get_markov()
    sp = int(os.environ.get("LOOPLING_MK_SP", "320"))
    nsg = int(os.environ.get("LOOPLING_MK_NSG", "4"))
    pval, pidx = p1(
        inputs=[latent.reshape(-1), fw_w, fw_s, fw_b, fold_b, base_row.reshape(-1)],
        template=[("VV", int(vocab)), ("SP", sp), ("NSG", nsg)],
        grid=(sp * 32 * nsg, 1, 1),
        threadgroup=(32 * nsg, 1, 1),
        output_shapes=[(sp,), (sp,)],
        output_dtypes=[mx.float32, mx.uint32],
    )
    return p2(
        inputs=[pval, pidx],
        template=[("SP", sp)],
        grid=(128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(1,)],
        output_dtypes=[mx.uint32],
    )[0]


def markov_bias_topk4_4b(latents, fw_w, fw_s, fw_b, fold_b, base_row, vocab, softcap):
    """Batched top-4 (tokens (M,4) u32, softmax probs (M,4) f32) for M in 1..3 latents sharing fw_w & base_row."""
    M = int(latents.shape[0])
    assert 1 <= M <= 3
    p1, p2 = _get_markov_topk(softcap)
    sp = int(os.environ.get("LOOPLING_MK_SP", "320"))
    nsg = int(os.environ.get("LOOPLING_MK_NSG", "4"))
    pval, pidx, pmax, psum = p1(
        inputs=[latents.reshape(-1), fw_w, fw_s, fw_b, fold_b, base_row.reshape(-1)],
        template=[("VV", int(vocab)), ("SP", sp), ("NSG", nsg), ("MM", M)],
        grid=(sp * 32 * nsg, 1, 1),
        threadgroup=(32 * nsg, 1, 1),
        output_shapes=[(M, sp, 4), (M, sp, 4), (M, sp), (M, sp)],
        output_dtypes=[mx.float32, mx.uint32, mx.float32, mx.float32],
    )
    return p2(
        inputs=[pval, pidx, pmax, psum],
        template=[("SP", sp)],
        grid=(M * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(M, 4), (M, 4)],
        output_dtypes=[mx.uint32, mx.float32],
    )


def _get_markov_beam2_p2():
    global _mk_beam2_p2_k
    if _mk_beam2_p2_k is None:
        _mk_beam2_p2_k = mx.fast.metal_kernel(
            name="markov_bias_topk4_beam2_4b_p2",
            input_names=["pval", "pidx", "pmax", "psum", "jp_prev", "pidx_prev", "st_d"],
            output_names=["ctok", "cjp", "cpar", "next_tok", "next_jp", "next_st", "next_pidx"],
            source=_MARKOV_BEAM2_P2_SRC,
            ensure_row_contiguous=True,
        )
    return _mk_beam2_p2_k


def markov_bias_topk4_beam_4b(latents, fw_w, fw_s, fw_b, fold_b, base_row, jp_prev, pidx_prev, st_d, vocab, softcap, pool_base):
    """Fused P1(MM=2) + single-TG beam-2 top-2 selection & RNN state gather on GPU."""
    assert int(latents.shape[0]) == 2 and int(st_d.shape[0]) == 2
    p1, _ = _get_markov_topk(softcap)
    sp = int(os.environ.get("LOOPLING_MK_SP", "320"))
    nsg = int(os.environ.get("LOOPLING_MK_NSG", "4"))
    pval, pidx, pmax, psum = p1(
        inputs=[latents.reshape(-1), fw_w, fw_s, fw_b, fold_b, base_row.reshape(-1)],
        template=[("VV", int(vocab)), ("SP", sp), ("NSG", nsg), ("MM", 2)],
        grid=(sp * 32 * nsg, 1, 1),
        threadgroup=(32 * nsg, 1, 1),
        output_shapes=[(2, sp, 4), (2, sp, 4), (2, sp), (2, sp)],
        output_dtypes=[mx.float32, mx.uint32, mx.float32, mx.float32],
    )
    rr = int(st_d.shape[1])
    return _get_markov_beam2_p2()(
        inputs=[pval, pidx, pmax, psum, jp_prev.reshape(-1), pidx_prev.reshape(-1), st_d.reshape(-1)],
        template=[("SP", sp), ("POOL_BASE", int(pool_base)), ("RR", rr)],
        grid=(256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(8,), (8,), (8,), (2,), (2,), (2, rr), (2,)],
        output_dtypes=[mx.uint32, mx.float32, mx.int32, mx.uint32, mx.float32, st_d.dtype, mx.int32],
    )


def markov_shortlist_topk4_beam_f16(latents, top_val, top_idx, fold_w_f16, jp_prev, pidx_prev, st_d, softcap, pool_base, temp=1.6):
    """Fused shortlist P1(MM=2) + beam-2 top-2 selection & RNN state gather on GPU."""
    assert int(latents.shape[0]) == 2 and int(st_d.shape[0]) == 2
    kk = int(top_val.shape[0])
    sp = 16 if kk <= 256 else 32
    nsg = 4
    p1 = _get_markov_sl_p1(softcap, 1.0 / float(temp))
    pval, pidx, pmax, psum = p1(
        inputs=[latents.reshape(-1), top_val.reshape(-1), top_idx.reshape(-1), fold_w_f16],
        template=[("KK", kk), ("SP", sp), ("NSG", nsg), ("MM", 2)],
        grid=(sp * 32 * nsg, 1, 1),
        threadgroup=(32 * nsg, 1, 1),
        output_shapes=[(2, sp, 4), (2, sp, 4), (2, sp), (2, sp)],
        output_dtypes=[mx.float32, mx.uint32, mx.float32, mx.float32],
    )
    rr = int(st_d.shape[1])
    return _get_markov_beam2_p2()(
        inputs=[pval, pidx, pmax, psum, jp_prev.reshape(-1), pidx_prev.reshape(-1), st_d.reshape(-1)],
        template=[("SP", sp), ("POOL_BASE", int(pool_base)), ("RR", rr)],
        grid=(256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(8,), (8,), (8,), (2,), (2,), (2, rr), (2,)],
        output_dtypes=[mx.uint32, mx.float32, mx.int32, mx.uint32, mx.float32, st_d.dtype, mx.int32],
    )


_TREE_WALK_COMMIT_SRC = r"""
  if (thread_position_in_threadgroup.x != 0) return;

  int path[8];
  int keep = 1;
  path[0] = 0;
  int cur_slot = 0;
  while (true) {
    uint target_tok = t[cur_slot];
    int matched = -1;
    for (int s = 1; s < WW; s++) {
      if (par[s] == cur_slot && blk[s] == target_tok) {
        matched = s;
        break;
      }
    }
    if (matched < 0) break;
    path[keep++] = matched;
    cur_slot = matched;
  }
  uint bonus_tok = t[cur_slot];
  for (int i = 0; i < WW; i++) {
    path_pad[i] = (i < keep) ? path[i] : 0;
    if (i < keep - 1) {
      emitted[i] = blk[path[i + 1]];
    } else if (i == keep - 1) {
      emitted[i] = bonus_tok;
    } else {
      emitted[i] = 0u;
    }
  }
  keep_arr[0] = keep;
  last_slot[0] = cur_slot;
  anchor_next[0] = bonus_tok;

  const int end0_r = meta_r[1];
  const int slot0_r = meta_r[2];
  const int cap_r = meta_r[3];
  const int max_size_r = meta_r[4];
  const int end1_r = end0_r + keep;
  const int s1_r = min(end1_r, max_size_r);
  const int next_slot0_r = (slot0_r + keep) % max_size_r;
  slot_r_old[0] = slot0_r;
  next_meta_r[0] = s1_r;
  next_meta_r[1] = end1_r;
  next_meta_r[2] = next_slot0_r;
  next_meta_r[3] = cap_r;
  next_meta_r[4] = max_size_r;
  next_meta_r[5] = 0;

  const int end0_f = meta_f[1];
  const int cap_f = meta_f[3];
  const int end1_f = end0_f + keep;
  end_f_old[0] = end0_f;
  next_meta_f[0] = end1_f;
  next_meta_f[1] = end1_f;
  next_meta_f[2] = 0;
  next_meta_f[3] = cap_f;
  next_meta_f[4] = 0;
  next_meta_f[5] = 0;

  if (META_LEN == 8) {
    const int st_old = meta_r[7];
    st_pos_old[0] = st_old;
    next_meta_r[6] = meta_r[6];
    next_meta_r[7] = st_old + keep;
    next_meta_f[6] = meta_f[6];
    next_meta_f[7] = st_old + keep;
  } else {
    st_pos_old[0] = 0;
  }

  const int P = dr_pos[0];
  dr_pos_old[0] = P;
  next_dr_pos[0] = P + keep;
"""


def tree_walk_commit_meta_gpu(t, blk, par, meta_r, meta_f, dr_pos, W, w_act=None):
    """On-GPU tree match walk + next KV/drafter position metadata update."""
    has_wa = w_act is not None
    k = _tree_walk_k.get(has_wa)
    if k is None:
        src = _TREE_WALK_COMMIT_SRC
        ins = ["t", "blk", "par", "meta_r", "meta_f", "dr_pos"]
        if has_wa:
            src = src.replace("for (int s = 1; s < WW; s++)", "int w_a = min(WW, max(1, w_act[0]));\n    for (int s = 1; s < w_a; s++)")
            assert src != _TREE_WALK_COMMIT_SRC
            ins.append("w_act")
        k = _tree_walk_k[has_wa] = mx.fast.metal_kernel(
            name="tree_walk_commit_meta_gpu_wa" if has_wa else "tree_walk_commit_meta_gpu",
            input_names=ins,
            output_names=[
                "path_pad", "emitted", "keep_arr", "last_slot", "anchor_next",
                "slot_r_old", "end_f_old", "dr_pos_old", "next_meta_r", "next_meta_f", "next_dr_pos",
                "st_pos_old",
            ],
            source=src,
            ensure_row_contiguous=True,
        )
    WW = int(W)
    assert 2 <= WW <= 8
    tu = t.reshape(-1)
    if tu.dtype != mx.uint32:
        tu = tu.astype(mx.uint32)
    bu = blk.reshape(-1)
    if bu.dtype != mx.uint32:
        bu = bu.astype(mx.uint32)
    pi = par.reshape(-1)
    if pi.dtype != mx.int32:
        pi = pi.astype(mx.int32)
    mr = meta_r.reshape(-1)
    mf = meta_f.reshape(-1)
    mlen = int(mr.shape[0])
    assert mlen in (6, 8) and int(mf.shape[0]) == mlen
    args = [tu, bu, pi, mr, mf, dr_pos.reshape(-1)] + ([w_act.reshape(-1)] if has_wa else [])
    res = k(
        inputs=args,
        template=[("WW", WW), ("META_LEN", mlen)],
        grid=(32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(WW,), (WW,), (1,), (1,), (1,), (1,), (1,), (1,), (mlen,), (mlen,), (1,), (1,)],
        output_dtypes=[
            mx.int32, mx.uint32, mx.int32, mx.int32, mx.uint32,
            mx.int32, mx.int32, mx.int32, mx.int32, mx.int32, mx.int32, mx.int32,
        ],
    )
    return res if mlen == 8 else res[:11]


_PACK_TREE_GPU_SRC = r"""
  if (thread_position_in_threadgroup.x != 0) return;

  uint tok[40];
  float jp[40];
  int par[40];
  int dep[40];
  for (int i = 0; i < PP; i++) {
    tok[i] = cand_toks[i];
    jp[i] = cand_jprobs[i];
    par[i] = cand_pars[i];
    dep[i] = (i < 4) ? 1 : (2 + (i - 4) / 8);
  }

  int ord[40];
  for (int i = 0; i < PP; i++) {
    int j = i;
    while (j > 0) {
      int prev = ord[j - 1];
      bool less = (jp[i] > jp[prev]) ||
                  (jp[i] == jp[prev] && (dep[i] < dep[prev] || (dep[i] == dep[prev] && i < prev)));
      if (!less) break;
      ord[j] = prev;
      j--;
    }
    ord[j] = i;
  }

  bool seen[41];
  seen[0] = true;
  for (int i = 1; i <= PP; i++) seen[i] = false;
  int sel_pool[8];
  int sel_dep[8];
  int sel_rank[8];
  int n_sel = 0;
  for (int k = 0; k < PP && n_sel < WW - 1; k++) {
    int pidx = ord[k] + 1;
    if (seen[pidx]) continue;
    int chain[8];
    int n_ch = 0;
    int cur = pidx;
    while (!seen[cur]) {
      chain[n_ch++] = cur;
      cur = par[cur - 1];
    }
    for (int c = n_ch - 1; c >= 0 && n_sel < WW - 1; c--) {
      int cp = chain[c];
      seen[cp] = true;
      sel_pool[n_sel] = cp;
      sel_dep[n_sel] = dep[cp - 1];
      sel_rank[n_sel] = n_sel;
      n_sel++;
    }
  }

  int topo_ord[8];
  for (int i = 0; i < WW - 1; i++) {
    int j = i;
    int di = sel_dep[i], ri = sel_rank[i];
    while (j > 0) {
      int prev = topo_ord[j - 1];
      if (di > sel_dep[prev] || (di == sel_dep[prev] && ri >= sel_rank[prev])) break;
      topo_ord[j] = prev;
      j--;
    }
    topo_ord[j] = i;
  }

  int p2s[41];
  for (int i = 0; i <= PP; i++) p2s[i] = 0;
  int local_par[8];
  int local_dep[8];
  block_toks[0] = anchor[0];
  parents_out[0] = -1;
  depths_out[0] = 0;
  local_par[0] = -1;
  local_dep[0] = 0;
  for (int s = 1; s < WW; s++) {
    int si = topo_ord[s - 1];
    int pidx = sel_pool[si];
    int ps = p2s[par[pidx - 1]];
    int db = sel_dep[si];
    p2s[pidx] = s;
    block_toks[s] = tok[pidx - 1];
    parents_out[s] = ps;
    depths_out[s] = db;
    local_par[s] = ps;
    local_dep[s] = db;
  }

  for (int b = 0; b < WW; b++) {
    int pb = local_par[b];
    int db = local_dep[b];
    int path[8];
    int cur = b;
    for (int k = db; k >= 0; k--) {
      path[k] = cur;
      cur = (k > 0) ? local_par[cur] : -1;
    }
    conv_anc[b * 3 + 0] = (db >= 3) ? path[db - 3] : -(db + 1);
    conv_anc[b * 3 + 1] = (db >= 2) ? path[db - 2] : -(db + 2);
    conv_anc[b * 3 + 2] = (db >= 1) ? pb : -3;
    tree_meta[b * 9 + 0] = db;
    for (int k = 0; k < 8; k++) {
      tree_meta[b * 9 + 1 + k] = (k <= db) ? path[k] : 0;
    }
  }
"""

_pack_tree_k = None


def pack_tree_gpu(cand_toks, cand_jprobs, cand_pars, anchor, W):
    """Zero-sync on-GPU dynamic tree packer -> (block_toks, parents_out, depths_out, conv_anc, tree_meta)."""
    global _pack_tree_k
    if _pack_tree_k is None:
        _pack_tree_k = mx.fast.metal_kernel(
            name="pack_dynamic_tree_gpu",
            input_names=["cand_toks", "cand_jprobs", "cand_pars", "anchor"],
            output_names=["block_toks", "parents_out", "depths_out", "conv_anc", "tree_meta"],
            source=_PACK_TREE_GPU_SRC,
            ensure_row_contiguous=True,
        )
    PP = int(cand_toks.shape[0])
    WW = int(W)
    assert 2 <= WW <= 8 and 1 <= WW - 1 <= PP <= 40
    return _pack_tree_k(
        inputs=[cand_toks.reshape(-1), cand_jprobs.reshape(-1), cand_pars.reshape(-1), anchor.reshape(-1)],
        template=[("PP", PP), ("WW", WW)],
        grid=(32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(WW,), (WW,), (WW,), (WW, 3), (WW, 9)],
        output_dtypes=[mx.uint32, mx.int32, mx.int32, mx.int32, mx.int32],
    )


_pack_tree_multi_k: dict = {}


def _pack_tree_multi_src(depth):
    PP = 4 + (depth - 1) * 8
    load = [
        "  if (thread_position_in_threadgroup.x != 0) return;",
        "  uint tok[40];",
        "  float jp[40];",
        "  int par[40];",
        "  int dep[40];",
        "  for (int i = 0; i < 4; i++) {",
        "    tok[i] = t0[i]; jp[i] = p0[i]; par[i] = 0; dep[i] = 1;",
        "  }",
    ]
    for d in range(1, depth):
        base = 4 + (d - 1) * 8
        load += [
            f"  for (int i = 0; i < 8; i++) {{",
            f"    tok[{base} + i] = t{d}[i]; jp[{base} + i] = p{d}[i]; par[{base} + i] = pr{d}[i]; dep[{base} + i] = {d + 1};",
            "  }",
        ]
    # Reuse the sort/select/pack tail starting at `_PACK_TREE_GPU_SRC` line `  int ord[40];`
    tail = _PACK_TREE_GPU_SRC.split("  int ord[40];", 1)[1]
    return "\n".join(load) + "\n  int ord[40];" + tail


def pack_tree_gpu_multi(tok0, prob0, depth_triples, anchor, W):
    """Zero-sync tree packer reading depth-0 `(4,)` and `(D-1)` beam triples directly without mx.concatenate."""
    D = 1 + len(depth_triples)
    assert 1 <= D <= 5
    PP = 4 + (D - 1) * 8
    WW = int(W)
    assert 2 <= WW <= 8 and 1 <= WW - 1 <= PP <= 40
    k = _pack_tree_multi_k.get(D)
    if k is None:
        ins = ["t0", "p0"]
        for d in range(1, D):
            ins += [f"t{d}", f"p{d}", f"pr{d}"]
        ins.append("anchor")
        k = _pack_tree_multi_k[D] = mx.fast.metal_kernel(
            name=f"pack_dynamic_tree_multi_d{D}",
            input_names=ins,
            output_names=["block_toks", "parents_out", "depths_out", "conv_anc", "tree_meta"],
            source=_pack_tree_multi_src(D),
            ensure_row_contiguous=True,
        )
    args = [tok0.reshape(-1), prob0.reshape(-1)]
    for ctok, cjp, cpar in depth_triples:
        args += [ctok.reshape(-1), cjp.reshape(-1), cpar.reshape(-1)]
    args.append(anchor.reshape(-1))
    return k(
        inputs=args,
        template=[("PP", PP), ("WW", WW)],
        grid=(32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(WW,), (WW,), (WW,), (WW, 3), (WW, 9)],
        output_dtypes=[mx.uint32, mx.int32, mx.int32, mx.int32, mx.int32],
    )


def _draft_markov_pack_src(scap_lit, inv_t_lit):
    pack_tail = _PACK_TREE_GPU_SRC.split("  int ord[40];", 1)[1]
    return f"""
  uint tid = thread_position_in_threadgroup.x; // 0..255 (rank R=256)
  uint lane = thread_index_in_simdgroup;
  uint sg = simdgroup_index_in_threadgroup;    // 0..7

  threadgroup float st_s[2][256];
  threadgroup float st_next[2][256];
  threadgroup float lat_s[2][256];
  threadgroup float w1_s[2][256];

  threadgroup uint cand_tok[40];
  threadgroup float cand_jp[40];
  threadgroup int cand_par[40];
  threadgroup int cand_dep[40];

  threadgroup float sg_max[2][8];
  threadgroup float sg_sum[2][8];
  threadgroup float sg_tv[2][8][4];
  threadgroup uint  sg_ti[2][8][4];

  threadgroup uint b_tok[2];
  threadgroup float b_jp[2];
  threadgroup int b_pidx[2];
  threadgroup uint tg_src[2];

  const float sc_cap = {scap_lit};
  const float inv_cap = 1.0f / {scap_lit};
  const float inv_t = {inv_t_lit};

  // --- Depth 0: RNN step from state=0 and w1[anchor] ---
  uint a0 = anchor[0];
  w1_s[0][tid] = float(w1[size_t(a0) * 256u + tid]);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  {{
    const device float* cx = ctx;
    float g0 = cx[tid], c0 = cx[256u + tid], l0 = cx[512u + tid];
    for (uint k = 0; k < 256u; k++) {{
      float v = w1_s[0][k];
      const device half* wr = w_recT + size_t(256u + k) * 768u;
      g0 = fma(v, float(wr[tid]), g0);
      c0 = fma(v, float(wr[256u + tid]), c0);
      l0 = fma(v, float(wr[512u + tid]), l0);
    }}
    float sg_f = 1.0f / (1.0f + metal::exp(-float(bfloat16_t(g0))));
    float cd_f = metal::tanh(float(bfloat16_t(c0)));
    float st_v = float(bfloat16_t(float(bfloat16_t(1.0f - sg_f)) * float(bfloat16_t(cd_f))));
    st_s[0][tid] = st_v;
    st_s[1][tid] = st_v;
    lat_s[0][tid] = float(bfloat16_t(metal::tanh(float(bfloat16_t(l0)))));
  }}
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Depth 0 shortlist scoring across 8 simdgroups
  {{
    float tv[4] = {{-INFINITY, -INFINITY, -INFINITY, -INFINITY}};
    uint  ti[4] = {{0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu}};
    float smax = -INFINITY, ssum = 0.0f;
    float4 l0 = ((const threadgroup float4*)(lat_s[0] + lane * 8u))[0];
    float4 l1 = ((const threadgroup float4*)(lat_s[0] + lane * 8u + 4u))[0];
    for (uint ci = sg; ci < KK; ci += 8u) {{
      uint tok_id = top_idx[ci];
      float bf = top_val[ci];
      uint4 wpk = ((const device uint4*)(fold_w + size_t(tok_id) * 256u + lane * 8u))[0];
      float4 w0 = float4(as_type<half4>(wpk.xy));
      float4 w1v = float4(as_type<half4>(wpk.zw));
      float val = simd_sum(dot(l0, w0) + dot(l1, w1v));
      if (lane == 0) {{
        float lg = sc_cap * fast::tanh((bf + val) * inv_cap) * inv_t;
        float nm = max(smax, lg);
        ssum = (smax > -INFINITY ? ssum * fast::exp(smax - nm) : 0.0f) + fast::exp(lg - nm);
        smax = nm;
        if (lg > tv[3] || (lg == tv[3] && tok_id < ti[3])) {{
          int pos = 3;
          while (pos > 0 && (lg > tv[pos - 1] || (lg == tv[pos - 1] && tok_id < ti[pos - 1]))) {{
            tv[pos] = tv[pos - 1]; ti[pos] = ti[pos - 1]; pos--;
          }}
          tv[pos] = lg; ti[pos] = tok_id;
        }}
      }}
    }}
    if (lane == 0) {{
      sg_max[0][sg] = smax;
      sg_sum[0][sg] = ssum;
      for (int k = 0; k < 4; k++) {{ sg_tv[0][sg][k] = tv[k]; sg_ti[0][sg][k] = ti[k]; }}
    }}
  }}
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0) {{
    float gm = -INFINITY, gs = 0.0f;
    for (int s = 0; s < 8; s++) {{
      float om = sg_max[0][s];
      if (om > -INFINITY) {{
        float nm = max(gm, om);
        gs = (gm > -INFINITY ? gs * fast::exp(gm - nm) : 0.0f) + sg_sum[0][s] * fast::exp(om - nm);
        gm = nm;
      }}
    }}
    float lse = gm + metal::log(gs);
    float fbv[4] = {{-INFINITY, -INFINITY, -INFINITY, -INFINITY}};
    uint  fbi[4] = {{0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu}};
    for (int s = 0; s < 8; s++) {{
      for (int k = 0; k < 4; k++) {{
        float v = sg_tv[0][s][k]; uint idx = sg_ti[0][s][k];
        if (v > fbv[3] || (v == fbv[3] && idx < fbi[3])) {{
          int pos = 3;
          while (pos > 0 && (v > fbv[pos - 1] || (v == fbv[pos - 1] && idx < fbi[pos - 1]))) {{
            fbv[pos] = fbv[pos - 1]; fbi[pos] = fbi[pos - 1]; pos--;
          }}
          fbv[pos] = v; fbi[pos] = idx;
        }}
      }}
    }}
    for (int k = 0; k < 4; k++) {{
      float prob = (fbv[k] > -INFINITY) ? fast::exp(fbv[k] - lse) : 0.0f;
      cand_tok[k] = fbi[k];
      cand_jp[k] = prob;
      cand_par[k] = 0;
      cand_dep[k] = 1;
    }}
    b_tok[0] = cand_tok[0]; b_tok[1] = cand_tok[1];
    b_jp[0] = cand_jp[0]; b_jp[1] = cand_jp[1];
    b_pidx[0] = 1; b_pidx[1] = 2;
  }}
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // --- Depths d = 1 .. DD - 1 (beam = 2) ---
  for (int d = 1; d < DD; d++) {{
    uint bt0 = b_tok[0], bt1 = b_tok[1];
    w1_s[0][tid] = float(w1[size_t(bt0) * 256u + tid]);
    w1_s[1][tid] = float(w1[size_t(bt1) * 256u + tid]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    {{
      const device float* cx = ctx + size_t(d) * 768u;
      for (int m = 0; m < 2; m++) {{
        float gd = cx[tid], cd = cx[256u + tid], ld = cx[512u + tid];
        for (uint k = 0; k < 256u; k++) {{
          float sv = st_s[m][k], wv = w1_s[m][k];
          const device half* wr0 = w_recT + size_t(k) * 768u;
          const device half* wr1 = w_recT + size_t(256u + k) * 768u;
          gd = fma(sv, float(wr0[tid]), fma(wv, float(wr1[tid]), gd));
          cd = fma(sv, float(wr0[256u + tid]), fma(wv, float(wr1[256u + tid]), cd));
          ld = fma(sv, float(wr0[512u + tid]), fma(wv, float(wr1[512u + tid]), ld));
        }}
        float sgd = 1.0f / (1.0f + metal::exp(-float(bfloat16_t(gd))));
        float cdd = metal::tanh(float(bfloat16_t(cd)));
        st_next[m][tid] = float(bfloat16_t(float(bfloat16_t(sgd)) * st_s[m][tid] + float(bfloat16_t(1.0f - sgd)) * float(bfloat16_t(cdd))));
        lat_s[m][tid] = float(bfloat16_t(metal::tanh(float(bfloat16_t(ld)))));
      }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    {{
      float tv[2][4]; uint ti[2][4]; float smax[2], ssum[2];
      for (int m = 0; m < 2; m++) {{
        for (int k = 0; k < 4; k++) {{ tv[m][k] = -INFINITY; ti[m][k] = 0xFFFFFFFFu; }}
        smax[m] = -INFINITY; ssum[m] = 0.0f;
      }}
      float4 l0_a = ((const threadgroup float4*)(lat_s[0] + lane * 8u))[0];
      float4 l1_a = ((const threadgroup float4*)(lat_s[0] + lane * 8u + 4u))[0];
      float4 l0_b = ((const threadgroup float4*)(lat_s[1] + lane * 8u))[0];
      float4 l1_b = ((const threadgroup float4*)(lat_s[1] + lane * 8u + 4u))[0];
      for (uint ci = sg; ci < KK; ci += 8u) {{
        size_t coi = size_t(d) * KK + ci;
        uint tok_id = top_idx[coi];
        float bf = top_val[coi];
        uint4 wpk = ((const device uint4*)(fold_w + size_t(tok_id) * 256u + lane * 8u))[0];
        float4 w0 = float4(as_type<half4>(wpk.xy));
        float4 w1v = float4(as_type<half4>(wpk.zw));
        float va = simd_sum(dot(l0_a, w0) + dot(l1_a, w1v));
        float vb = simd_sum(dot(l0_b, w0) + dot(l1_b, w1v));
        if (lane == 0) {{
          float vs[2] = {{va, vb}};
          for (int m = 0; m < 2; m++) {{
            float lg = sc_cap * fast::tanh((bf + vs[m]) * inv_cap) * inv_t;
            float nm = max(smax[m], lg);
            ssum[m] = (smax[m] > -INFINITY ? ssum[m] * fast::exp(smax[m] - nm) : 0.0f) + fast::exp(lg - nm);
            smax[m] = nm;
            if (lg > tv[m][3] || (lg == tv[m][3] && tok_id < ti[m][3])) {{
              int pos = 3;
              while (pos > 0 && (lg > tv[m][pos - 1] || (lg == tv[m][pos - 1] && tok_id < ti[m][pos - 1]))) {{
                tv[m][pos] = tv[m][pos - 1]; ti[m][pos] = ti[m][pos - 1]; pos--;
              }}
              tv[m][pos] = lg; ti[m][pos] = tok_id;
            }}
          }}
        }}
      }}
      if (lane == 0) {{
        for (int m = 0; m < 2; m++) {{
          sg_max[m][sg] = smax[m];
          sg_sum[m][sg] = ssum[m];
          for (int k = 0; k < 4; k++) {{ sg_tv[m][sg][k] = tv[m][k]; sg_ti[m][sg][k] = ti[m][k]; }}
        }}
      }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {{
      float all_jp[8];
      uint all_tok[8];
      int pool_base = 1 + 4 + (d - 1) * 8;
      int base = 4 + (d - 1) * 8;
      for (int m = 0; m < 2; m++) {{
        float gm = -INFINITY, gs = 0.0f;
        for (int s = 0; s < 8; s++) {{
          float om = sg_max[m][s];
          if (om > -INFINITY) {{
            float nm = max(gm, om);
            gs = (gm > -INFINITY ? gs * fast::exp(gm - nm) : 0.0f) + sg_sum[m][s] * fast::exp(om - nm);
            gm = nm;
          }}
        }}
        float lse = gm + metal::log(gs);
        float fbv[4] = {{-INFINITY, -INFINITY, -INFINITY, -INFINITY}};
        uint  fbi[4] = {{0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu}};
        for (int s = 0; s < 8; s++) {{
          for (int k = 0; k < 4; k++) {{
            float v = sg_tv[m][s][k]; uint idx = sg_ti[m][s][k];
            if (v > fbv[3] || (v == fbv[3] && idx < fbi[3])) {{
              int pos = 3;
              while (pos > 0 && (v > fbv[pos - 1] || (v == fbv[pos - 1] && idx < fbi[pos - 1]))) {{
                fbv[pos] = fbv[pos - 1]; fbi[pos] = fbi[pos - 1]; pos--;
              }}
              fbv[pos] = v; fbi[pos] = idx;
            }}
          }}
        }}
        float jpp = b_jp[m];
        int parp = b_pidx[m];
        for (int k = 0; k < 4; k++) {{
          float prob = (fbv[k] > -INFINITY) ? fast::exp(fbv[k] - lse) : 0.0f;
          float jpc = jpp * prob;
          int gi = m * 4 + k;
          all_jp[gi] = jpc;
          all_tok[gi] = fbi[k];
          cand_tok[base + gi] = fbi[k];
          cand_jp[base + gi] = jpc;
          cand_par[base + gi] = parp;
          cand_dep[base + gi] = d + 1;
        }}
      }}
      float v0 = -1.0f, v1 = -1.0f;
      uint b0 = 0, b1 = 1;
      for (uint i = 0; i < 8; i++) {{
        float v = all_jp[i];
        if (v > v0) {{ v1 = v0; b1 = b0; v0 = v; b0 = i; }}
        else if (v > v1) {{ v1 = v; b1 = i; }}
      }}
      b_tok[0] = all_tok[b0]; b_tok[1] = all_tok[b1];
      b_jp[0] = v0; b_jp[1] = v1;
      b_pidx[0] = pool_base + int(b0); b_pidx[1] = pool_base + int(b1);
      tg_src[0] = b0 >> 2; tg_src[1] = b1 >> 2;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (d + 1 < DD) {{
      uint s0 = tg_src[0], s1 = tg_src[1];
      st_s[0][tid] = st_next[s0][tid];
      st_s[1][tid] = st_next[s1][tid];
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
  }}

  if (tid != 0) return;
  {{
    uint tok[40]; float jp[40]; int par[40]; int dep[40];
    for (int i = 0; i < PP; i++) {{
      tok[i] = cand_tok[i]; jp[i] = cand_jp[i]; par[i] = cand_par[i]; dep[i] = cand_dep[i];
    }}
    int ord[40];{pack_tail}
  }}
"""


def draft_markov_pack_gpu(ctx, w1, w_recT_f16, top_val, top_idx, fold_w_f16, anchor, max_depth, W, softcap, temp=1.6):
    """Single-TG kernel fusing all DD RNN steps + beam-2 shortlist Markov + dynamic W-node tree pack."""
    DD = int(max_depth)
    WW = int(W)
    PP = 4 + (DD - 1) * 8
    assert 1 <= DD <= 5 and 2 <= WW <= 8 and 1 <= WW - 1 <= PP <= 40
    kk = int(top_val.shape[1])
    inv_t = 1.0 / float(temp)
    key = (float(softcap), inv_t)
    k = _draft_mk_pack_k.get(key)
    if k is None:
        scap_lit = flit(float(softcap))
        inv_t_lit = flit(inv_t)
        sfx = f"{scap_lit}_{inv_t_lit}".replace(".", "p").replace("-", "m").replace("+", "P")
        k = _draft_mk_pack_k[key] = mx.fast.metal_kernel(
            name=f"draft_markov_pack_gpu_{sfx}",
            input_names=["ctx", "w1", "w_recT", "top_val", "top_idx", "fold_w", "anchor"],
            output_names=["block_toks", "parents_out", "depths_out", "conv_anc", "tree_meta"],
            source=_draft_markov_pack_src(scap_lit, inv_t_lit),
            ensure_row_contiguous=True,
        )
    return k(
        inputs=[ctx[:DD].reshape(-1), w1, w_recT_f16, top_val[:DD].reshape(-1), top_idx[:DD].reshape(-1), fold_w_f16, anchor.reshape(-1)],
        template=[("DD", DD), ("KK", kk), ("WW", WW), ("PP", PP)],
        grid=(256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(WW,), (WW,), (WW,), (WW, 3), (WW, 9)],
        output_dtypes=[mx.uint32, mx.int32, mx.int32, mx.int32, mx.int32],
    )


def _pack_tree_multi_dink_src(depth):
    load = [
        "  if (thread_position_in_threadgroup.x != 0) return;",
        "  uint tok[40];",
        "  float jp[40];",
        "  int par[40];",
        "  int dep[40];",
        "  for (int i = 0; i < 4; i++) {",
        "    tok[i] = t0[i]; jp[i] = p0[i]; par[i] = 0; dep[i] = 1;",
        "  }",
    ]
    for d in range(1, depth):
        base = 4 + (d - 1) * 8
        load += [
            "  for (int i = 0; i < 8; i++) {",
            f"    tok[{base} + i] = t{d}[i]; jp[{base} + i] = p{d}[i]; par[{base} + i] = pr{d}[i]; dep[{base} + i] = {d + 1};",
            "  }",
        ]
    body = r"""
  int ord[40];
  for (int i = 0; i < PP; i++) {
    int j = i;
    while (j > 0) {
      int prev = ord[j - 1];
      bool less = (jp[i] > jp[prev]) ||
                  (jp[i] == jp[prev] && (dep[i] < dep[prev] || (dep[i] == dep[prev] && i < prev)));
      if (!less) break;
      ord[j] = prev;
      j--;
    }
    ord[j] = i;
  }

  bool seen[41];
  seen[0] = true;
  for (int i = 1; i <= PP; i++) seen[i] = false;
  int sel_pool[8];
  int sel_dep[8];
  int sel_rank[8];
  float sel_jp[8];
  int n_sel = 0;
  for (int k = 0; k < PP && n_sel < WW - 1; k++) {
    int pidx = ord[k] + 1;
    if (seen[pidx]) continue;
    int chain[8];
    int n_ch = 0;
    int cur = pidx;
    while (!seen[cur]) {
      chain[n_ch++] = cur;
      cur = par[cur - 1];
    }
    for (int c = n_ch - 1; c >= 0 && n_sel < WW - 1; c--) {
      int cp = chain[c];
      seen[cp] = true;
      sel_pool[n_sel] = cp;
      sel_dep[n_sel] = dep[cp - 1];
      sel_rank[n_sel] = n_sel;
      sel_jp[n_sel] = jp[cp - 1];
      n_sel++;
    }
  }

  int fw = int(dink_cfg[1] + 0.5f);
  float cal_s = dink_cfg[0];
  int best_w = WW;
  if (fw > 0) {
    best_w = min(WW, max(2, fw));
  } else {
    float best_r = -1.0f;
    float acc = 0.0f;
    for (int w = 2; w <= WW; w++) {
      acc += sel_jp[w - 2];
      float r = (1.0f + cal_s * acc) / cost_arr[w];
      if (r > best_r) {
        best_r = r;
        best_w = w;
      }
    }
  }
  w_act_out[0] = best_w;

  int n_act = best_w - 1;
  int topo_ord[8];
  for (int i = 0; i < n_act; i++) {
    int j = i;
    int di = sel_dep[i], ri = sel_rank[i];
    while (j > 0) {
      int prev = topo_ord[j - 1];
      if (di > sel_dep[prev] || (di == sel_dep[prev] && ri >= sel_rank[prev])) break;
      topo_ord[j] = prev;
      j--;
    }
    topo_ord[j] = i;
  }

  int p2s[41];
  for (int i = 0; i <= PP; i++) p2s[i] = 0;
  int local_par[8];
  int local_dep[8];
  uint anc_tok = anchor[0];
  block_toks[0] = anc_tok;
  parents_out[0] = -1;
  depths_out[0] = 0;
  local_par[0] = -1;
  local_dep[0] = 0;
  for (int s = 1; s < best_w; s++) {
    int si = topo_ord[s - 1];
    int pidx = sel_pool[si];
    int ps = p2s[par[pidx - 1]];
    int db = sel_dep[si];
    p2s[pidx] = s;
    block_toks[s] = tok[pidx - 1];
    parents_out[s] = ps;
    depths_out[s] = db;
    local_par[s] = ps;
    local_dep[s] = db;
  }
  for (int s = best_w; s < WW; s++) {
    block_toks[s] = anc_tok;
    parents_out[s] = 0;
    depths_out[s] = 1;
    local_par[s] = 0;
    local_dep[s] = 1;
  }

  for (int b = 0; b < WW; b++) {
    int pb = local_par[b];
    int db = local_dep[b];
    int path[8];
    int cur = b;
    for (int k = db; k >= 0; k--) {
      path[k] = cur;
      cur = (k > 0) ? local_par[cur] : -1;
    }
    conv_anc[b * 3 + 0] = (db >= 3) ? path[db - 3] : -(db + 1);
    conv_anc[b * 3 + 1] = (db >= 2) ? path[db - 2] : -(db + 2);
    conv_anc[b * 3 + 2] = (db >= 1) ? pb : -3;
    tree_meta[b * 9 + 0] = db;
    for (int k = 0; k < 8; k++) {
      tree_meta[b * 9 + 1 + k] = (k <= db) ? path[k] : 0;
    }
  }
"""
    return "\n".join(load) + body


def pack_tree_gpu_multi_dink(tok0, prob0, depth_triples, anchor, W_max, cost_arr, dink_cfg):
    """Zero-sync Beam-2 tree packer with on-GPU Dinkelbach verify-width selection `w_act in [2, W_max]`."""
    D = 1 + len(depth_triples)
    assert 1 <= D <= 5
    PP = 4 + (D - 1) * 8
    WW = int(W_max)
    assert 2 <= WW <= 8 and 1 <= WW - 1 <= PP <= 40
    k = _pack_tree_multi_dink_k.get(D)
    if k is None:
        ins = ["t0", "p0"]
        for d in range(1, D):
            ins += [f"t{d}", f"p{d}", f"pr{d}"]
        ins += ["anchor", "cost_arr", "dink_cfg"]
        k = _pack_tree_multi_dink_k[D] = mx.fast.metal_kernel(
            name=f"pack_dynamic_tree_multi_dink_d{D}",
            input_names=ins,
            output_names=["block_toks", "parents_out", "depths_out", "conv_anc", "tree_meta", "w_act_out"],
            source=_pack_tree_multi_dink_src(D),
            ensure_row_contiguous=True,
        )
    anc_u = anchor.reshape(-1)
    if anc_u.dtype != mx.uint32:
        anc_u = anc_u.astype(mx.uint32)
    args = [tok0.reshape(-1), prob0.reshape(-1)]
    for ctok, cjp, cpar in depth_triples:
        args += [ctok.reshape(-1), cjp.reshape(-1), cpar.reshape(-1)]
    args += [anc_u, cost_arr.reshape(-1), dink_cfg.reshape(-1)]
    return k(
        inputs=args,
        template=[("PP", PP), ("WW", WW)],
        grid=(32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(WW,), (WW,), (WW,), (WW, 3), (WW, 9), (1,)],
        output_dtypes=[mx.uint32, mx.int32, mx.int32, mx.int32, mx.int32, mx.int32],
    )


# ---------------------------------------------------------------------------------------------------- DFlash2 BF16 RO Kernels
def _qmv4_bf16_ro_src(swiglu: bool = False):
    if not swiglu:
        epi = r"""
  _Pragma("clang loop unroll(full)")
  for (int b = 0; b < RB; b++) {
    _Pragma("clang loop unroll(full)")
    for (int row = 0; row < RPS; row++) {
      float r = simd_sum(result[b][row]);
      if (simd_lid == 0 && row <= last_row && (FULL || b < nb)) {
        yp[size_t(b) * N + row] = static_cast<bfloat16_t>(r);
      }
    }
  }
"""
        y_ptr = "  device bfloat16_t* yp = y + size_t(b0) * N + out_row;"
    else:
        epi = r"""
  _Pragma("clang loop unroll(full)")
  for (int b = 0; b < RB; b++) {
    _Pragma("clang loop unroll(full)")
    for (int pair = 0; pair < RPS / 2; pair++) {
      float rg = simd_sum(result[b][2 * pair]);
      float ru = simd_sum(result[b][2 * pair + 1]);
      if (simd_lid == 0 && (2 * pair + 1) <= last_row && (FULL || b < nb)) {
        float g = float(static_cast<bfloat16_t>(rg));
        float u = float(static_cast<bfloat16_t>(ru));
        float act = (g / (1.0f + metal::exp(-g))) * u;
        yp[size_t(b) * (N / 2) + pair] = static_cast<bfloat16_t>(act);
      }
    }
  }
"""
        y_ptr = "  device bfloat16_t* yp = y + size_t(b0) * (N / 2) + (out_row / 2);"
    return r"""
  constexpr int group_size = 64;
  constexpr int block_size = 256;
  static_assert(K % block_size == 0, "qmv4 bf16 ro needs K % 256 == 0");
  constexpr bool FULL = (B % RB) == 0;

  uint ty = threadgroup_position_in_grid.y;
  uint b0 = threadgroup_position_in_grid.z * RB;
  uint simd_gid = simdgroup_index_in_threadgroup;
  uint simd_lid = thread_index_in_simdgroup;

  const int out_row = int(ty) * (NSG * RPS) + int(simd_gid) * RPS;
  int last_row = RPS - 1;
  if (PR) {
    if (out_row >= N) return;
    last_row = min(last_row, N - 1 - out_row);
  }

  const int nb = FULL ? RB : min(RB, B - int(b0));
  thread float result[RB][RPS];
  for (int b = 0; b < RB; b++)
    for (int r = 0; r < RPS; r++) result[b][r] = 0.0f;

  const int in_vec_size_w = K / 2;
  const int in_vec_size_g = K / group_size;

  const device uint8_t* ws = (const device uint8_t*)w + size_t(out_row) * in_vec_size_w + simd_lid * 4;
  const device bfloat16_t* sc_p = scales + size_t(out_row) * in_vec_size_g + (simd_lid >> 3);
  const device bfloat16_t* bi_p = biases + size_t(out_row) * in_vec_size_g + (simd_lid >> 3);
  const device bfloat16_t* xp0 = x + size_t(b0) * K + simd_lid * 8;
""" + y_ptr + r"""

  for (int k = 0; k < K; k += block_size) {
    _Pragma("clang loop unroll(full)")
    for (int row = 0; row < RPS; row++) {
      int src = PR ? min(row, last_row) : row;
      uint32_t wv = ((const device uint32_t*)(ws + size_t(src) * in_vec_size_w))[0];
      float sc = float(sc_p[size_t(src) * in_vec_size_g]);
      float bi = float(bi_p[size_t(src) * in_vec_size_g]);
      float4 w0 = float4(float(wv & 0xFu), float((wv >> 4) & 0xFu), float((wv >> 8) & 0xFu), float((wv >> 12) & 0xFu)) * sc + bi;
      float4 w1 = float4(float((wv >> 16) & 0xFu), float((wv >> 20) & 0xFu), float((wv >> 24) & 0xFu), float((wv >> 28) & 0xFu)) * sc + bi;
      _Pragma("clang loop unroll(full)")
      for (int b = 0; b < RB; b++) {
        if (FULL || b < nb) {
          uint4 xpk = ((const device uint4*)(xp0 + size_t(b) * K))[0];
          float4 xv0 = float4(as_type<bfloat4>(xpk.xy));
          float4 xv1 = float4(as_type<bfloat4>(xpk.zw));
          result[b][row] += dot(xv0, w0) + dot(xv1, w1);
        }
      }
    }
    ws += 128;
    sc_p += 4;
    bi_p += 4;
    xp0 += block_size;
  }
""" + epi


def _run_qmv4_bf16_ro(x: mx.array, w: mx.array, scales: mx.array, biases: mx.array, swiglu: bool = False) -> mx.array:
    B, K = int(x.shape[0]), int(x.shape[1])
    N = int(w.shape[0])
    assert x.ndim == 2 and x.dtype == mx.bfloat16 and scales.dtype == mx.bfloat16 and biases.dtype == mx.bfloat16
    assert 1 <= B <= MAX_B and K % 256 == 0 and (not swiglu or N % 2 == 0)
    kfn = _qmv4_bf16_ro_k.get(swiglu)
    if kfn is None:
        kfn = _qmv4_bf16_ro_k[swiglu] = mx.fast.metal_kernel(
            name="rows_qmv4_bf16_ro_sw" if swiglu else "rows_qmv4_bf16_ro",
            input_names=["x", "w", "scales", "biases"],
            output_names=["y"],
            source=_qmv4_bf16_ro_src(swiglu=swiglu),
            ensure_row_contiguous=True,
        )
    ov = os.environ.get("LOOPLING_QMV4_BF16_CFG")
    if ov:
        rps, nsg, rb = (int(t) for t in ov.split(","))
        rb = min(rb, B)
    elif B == 1:
        rps, nsg, rb = (2, 2, 1)
    elif N <= 1024:
        rps, nsg, rb = (2, 2, min(8, B))
    else:
        rps, nsg, rb = (2, 4, min(8, B))
    tiles = (N + rps * nsg - 1) // (rps * nsg)
    pr = (N % (rps * nsg)) != 0
    out_n = (N // 2) if swiglu else N
    return kfn(
        inputs=[x, w, scales, biases],
        template=[("B", B), ("K", K), ("N", N), ("PR", pr), ("RPS", rps), ("NSG", nsg), ("RB", rb)],
        grid=(32, nsg * tiles, (B + rb - 1) // rb),
        threadgroup=(32, nsg, 1),
        output_shapes=[(B, out_n)],
        output_dtypes=[mx.bfloat16],
    )[0]


def qmv4_bf16_ro_rows(x: mx.array, w: mx.array, scales: mx.array, biases: mx.array) -> mx.array:
    """Register-lean row-outer 4-bit bf16 matvec (RPS=2, zero spills at B=8)."""
    return _run_qmv4_bf16_ro(x, w, scales, biases, swiglu=False)


def qmv4_bf16_swiglu_ro_rows(x: mx.array, w: mx.array, scales: mx.array, biases: mx.array) -> mx.array:
    """Register-lean row-outer 4-bit bf16 interleaved gate/up + SwiGLU (RPS=2, zero spills at B=8)."""
    return _run_qmv4_bf16_ro(x, w, scales, biases, swiglu=True)


def _qmv4_bf16_u2_src(swiglu_b8: bool = False):
    if not swiglu_b8:
        row_setup = r"""
  const int out_row = int(ty) * (NSG * RPS) + int(simd_gid) * RPS;
  if (out_row >= N) return;
  const int last_row = min(RPS - 1, N - 1 - out_row);
"""
        load_full = r"""
    _Pragma("clang loop unroll(full)")
    for (int row = 0; row < RPS; row++) {
      int src = min(row, last_row);
      wr[row] = ((const device uint2*)(ws + size_t(src) * in_w))[0];
      sc[row] = float(sc_p[size_t(src) * in_g]);
      bi[row] = float(bi_p[size_t(src) * in_g]);
    }
"""
        load_tail = r"""
    _Pragma("clang loop unroll(full)")
    for (int row = 0; row < RPS; row++) {
      int src = min(row, last_row);
      wr[row] = ((const device uint32_t*)(ws_t + size_t(src) * in_w))[0];
      sc[row] = float(sc_t[size_t(src) * in_g]);
      bi[row] = float(bi_t[size_t(src) * in_g]);
    }
"""
        epi = r"""
  _Pragma("clang loop unroll(full)")
  for (int b = 0; b < BB; b++) {
    _Pragma("clang loop unroll(full)")
    for (int row = 0; row < RPS; row++) {
      float r = simd_sum(result[b][row]);
      if (simd_lid == 0 && row <= last_row) {
        y[size_t(b) * N + size_t(out_row + row)] = static_cast<bfloat16_t>(r);
      }
    }
  }
"""
    else:
        # Block-8 interleaved [g(0..7), u(0..7), ...]; RPS == 2 (row 0 = gate, row 1 = up)
        row_setup = r"""
  const int out_half_n = N >> 1;
  const int pair = int(ty) * NSG + int(simd_gid);
  if (pair >= out_half_n) return;
  const int out_row = ((pair >> 3) << 4) + (pair & 7);
"""
        load_full = r"""
    wr[0] = ((const device uint2*)ws)[0];
    sc[0] = float(sc_p[0]);
    bi[0] = float(bi_p[0]);
    wr[1] = ((const device uint2*)(ws + size_t(8) * in_w))[0];
    sc[1] = float(sc_p[size_t(8) * in_g]);
    bi[1] = float(bi_p[size_t(8) * in_g]);
"""
        load_tail = r"""
    wr[0] = ((const device uint32_t*)ws_t)[0];
    sc[0] = float(sc_t[0]);
    bi[0] = float(bi_t[0]);
    wr[1] = ((const device uint32_t*)(ws_t + size_t(8) * in_w))[0];
    sc[1] = float(sc_t[size_t(8) * in_g]);
    bi[1] = float(bi_t[size_t(8) * in_g]);
"""
        epi = r"""
  _Pragma("clang loop unroll(full)")
  for (int b = 0; b < BB; b++) {
    float rg = simd_sum(result[b][0]);
    float ru = simd_sum(result[b][1]);
    if (simd_lid == 0) {
      float g = float(static_cast<bfloat16_t>(rg));
      float u = float(static_cast<bfloat16_t>(ru));
      float act = (g / (1.0f + metal::exp(-g))) * u;
      y[size_t(b) * out_half_n + size_t(pair)] = static_cast<bfloat16_t>(act);
    }
  }
"""
    return (
        r"""
  constexpr int group_size = 64;
  constexpr int NFULL = K / 512;
  constexpr int TAIL = K % 512;
  static_assert(K % 256 == 0, "qmv4 bf16 u2 needs K % 256 == 0");

  uint ty = threadgroup_position_in_grid.y;
  uint simd_gid = simdgroup_index_in_threadgroup;
  uint simd_lid = thread_index_in_simdgroup;
"""
        + row_setup
        + r"""
  thread float result[BB][RPS];
  _Pragma("clang loop unroll(full)")
  for (int b = 0; b < BB; b++)
    _Pragma("clang loop unroll(full)")
    for (int r = 0; r < RPS; r++) result[b][r] = 0.0f;

  const int in_w = K / 2;
  const int in_g = K / group_size;
  const float4 inv_s = float4(1.0f, 1.0f / 16.0f, 1.0f / 256.0f, 1.0f / 4096.0f);

  const device uint8_t* ws = (const device uint8_t*)w + size_t(out_row) * in_w + simd_lid * 8;
  const device bfloat16_t* sc_p = scales + size_t(out_row) * in_g + (simd_lid >> 2);
  const device bfloat16_t* bi_p = biases + size_t(out_row) * in_g + (simd_lid >> 2);
  const device bfloat16_t* xp = x + simd_lid * 16;

  for (int it = 0; it < NFULL; it++) {
    thread uint2 wr[RPS];
    thread float sc[RPS], bi[RPS];
"""
        + load_full
        + r"""
    _Pragma("clang loop unroll(full)")
    for (int b = 0; b < BB; b++) {
      const device bfloat16_t* xb = xp + size_t(b) * K;
      uint4 xpk0 = ((const device uint4*)xb)[0];
      uint4 xpk1 = ((const device uint4*)(xb + 8))[0];
      float4 a0 = float4(as_type<bfloat4>(xpk0.xy));
      float4 a1 = float4(as_type<bfloat4>(xpk0.zw));
      float4 a2 = float4(as_type<bfloat4>(xpk1.xy));
      float4 a3 = float4(as_type<bfloat4>(xpk1.zw));
      float smb = ((a0.x + a0.y + a0.z + a0.w) + (a1.x + a1.y + a1.z + a1.w))
                + ((a2.x + a2.y + a2.z + a2.w) + (a3.x + a3.y + a3.z + a3.w));
      float4 xf0 = a0 * inv_s, xf1 = a1 * inv_s, xf2 = a2 * inv_s, xf3 = a3 * inv_s;
      _Pragma("clang loop unroll(full)")
      for (int row = 0; row < RPS; row++) {
        uint2 wv = wr[row];
        ushort u0 = ushort(wv.x & 0xFFFFu), u1 = ushort(wv.x >> 16);
        ushort u2 = ushort(wv.y & 0xFFFFu), u3 = ushort(wv.y >> 16);
        float4 w0 = float4(float(u0 & 0x000Fu), float(u0 & 0x00F0u), float(u0 & 0x0F00u), float(u0 & 0xF000u));
        float4 w1 = float4(float(u1 & 0x000Fu), float(u1 & 0x00F0u), float(u1 & 0x0F00u), float(u1 & 0xF000u));
        float4 w2 = float4(float(u2 & 0x000Fu), float(u2 & 0x00F0u), float(u2 & 0x0F00u), float(u2 & 0xF000u));
        float4 w3 = float4(float(u3 & 0x000Fu), float(u3 & 0x00F0u), float(u3 & 0x0F00u), float(u3 & 0xF000u));
        float dp = (dot(xf0, w0) + dot(xf1, w1)) + (dot(xf2, w2) + dot(xf3, w3));
        result[b][row] += sc[row] * dp + smb * bi[row];
      }
    }
    ws += 256;
    sc_p += 8;
    bi_p += 8;
    xp += 512;
  }

  if (TAIL == 256) {
    const device uint8_t* ws_t = (const device uint8_t*)w + size_t(out_row) * in_w + NFULL * 256 + simd_lid * 4;
    const device bfloat16_t* sc_t = scales + size_t(out_row) * in_g + NFULL * 8 + (simd_lid >> 3);
    const device bfloat16_t* bi_t = biases + size_t(out_row) * in_g + NFULL * 8 + (simd_lid >> 3);
    const device bfloat16_t* xp_t = x + NFULL * 512 + simd_lid * 8;
    thread uint32_t wr[RPS];
    thread float sc[RPS], bi[RPS];
"""
        + load_tail
        + r"""
    _Pragma("clang loop unroll(full)")
    for (int b = 0; b < BB; b++) {
      uint4 xpk = ((const device uint4*)(xp_t + size_t(b) * K))[0];
      float4 a0 = float4(as_type<bfloat4>(xpk.xy));
      float4 a1 = float4(as_type<bfloat4>(xpk.zw));
      float smb = (a0.x + a0.y + a0.z + a0.w) + (a1.x + a1.y + a1.z + a1.w);
      float4 xf0 = a0 * inv_s, xf1 = a1 * inv_s;
      _Pragma("clang loop unroll(full)")
      for (int row = 0; row < RPS; row++) {
        uint32_t wv = wr[row];
        ushort u0 = ushort(wv & 0xFFFFu), u1 = ushort(wv >> 16);
        float4 w0 = float4(float(u0 & 0x000Fu), float(u0 & 0x00F0u), float(u0 & 0x0F00u), float(u0 & 0xF000u));
        float4 w1 = float4(float(u1 & 0x000Fu), float(u1 & 0x00F0u), float(u1 & 0x0F00u), float(u1 & 0xF000u));
        result[b][row] += sc[row] * (dot(xf0, w0) + dot(xf1, w1)) + smb * bi[row];
      }
    }
  }
"""
        + epi
    )


def _run_qmv4_bf16_u2(x: mx.array, w: mx.array, scales: mx.array, biases: mx.array, swiglu_b8: bool = False) -> mx.array:
    BB, K = int(x.shape[0]), int(x.shape[1])
    N = int(w.shape[0])
    assert 1 <= BB <= 8 and K % 256 == 0 and x.dtype == mx.bfloat16 and scales.dtype == mx.bfloat16 and biases.dtype == mx.bfloat16
    assert (not swiglu_b8) or (N % 16 == 0)
    kfn = _qmv4_bf16_u2_k.get(swiglu_b8)
    if kfn is None:
        kfn = _qmv4_bf16_u2_k[swiglu_b8] = mx.fast.metal_kernel(
            name="rows_qmv4_bf16_u2_sw8" if swiglu_b8 else "rows_qmv4_bf16_u2",
            input_names=["x", "w", "scales", "biases"],
            output_names=["y"],
            source=_qmv4_bf16_u2_src(swiglu_b8=swiglu_b8),
            ensure_row_contiguous=True,
        )
    ov = os.environ.get("LOOPLING_QMV4_BF16_U2_CFG")
    if swiglu_b8:
        nsg = int(ov.split(",")[0]) if ov else 2
        rps = 2
        out_n = N >> 1
        tiles = (out_n + nsg - 1) // nsg
    else:
        if ov:
            nsg, rps = (int(t) for t in ov.split(","))
        else:
            nsg, rps = 2, 2
        out_n = N
        bn = nsg * rps
        tiles = (N + bn - 1) // bn
    return kfn(
        inputs=[x, w, scales, biases],
        template=[("BB", BB), ("K", K), ("N", N), ("NSG", nsg), ("RPS", rps)],
        grid=(32, nsg * tiles, 1),
        threadgroup=(32, nsg, 1),
        output_shapes=[(BB, out_n)],
        output_dtypes=[mx.bfloat16],
    )[0]


def qmv4_bf16_u2_rows(x: mx.array, w: mx.array, scales: mx.array, biases: mx.array) -> mx.array:
    """64-bit uint2 (512-elem/iter) barrier-free 4-bit bf16 matvec for B<=8."""
    return _run_qmv4_bf16_u2(x, w, scales, biases, swiglu_b8=False)


def qmv4_bf16_u2_swiglu_b8_rows(x: mx.array, w: mx.array, scales: mx.array, biases: mx.array) -> mx.array:
    """64-bit uint2 (512-elem/iter) block-8 interleaved gate/up + SwiGLU for B<=8."""
    return _run_qmv4_bf16_u2(x, w, scales, biases, swiglu_b8=True)


def _qmv4_bf16_mma8_src(swiglu: bool = False):
    if not swiglu:
        epi = r"""
  threadgroup float* out_s = (threadgroup float*)sw_sg;
  _Pragma("clang loop unroll(full)")
  for (int rt = 0; rt < RPT; rt++) {
    simdgroup_store(acc[rt], out_s, 8);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    int r_base = sg_row + rt * 8;
    for (uint i = lane; i < 64u; i += 32u) {
      uint rb = i >> 3;
      uint rn = i & 7u;
      if (r_base + int(rn) < N) {
        y[size_t(rb) * N + size_t(r_base + int(rn))] = static_cast<bfloat16_t>(out_s[i]);
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
"""
    else:
        # Weights packed in 8-row gate/up blocks: [g(0..7), u(0..7), g(8..15), u(8..15), ...]. RPT == 2 per SG!
        epi = r"""
  threadgroup float* out_g = (threadgroup float*)sw_sg;
  threadgroup float* out_u = out_g + 64;
  simdgroup_store(acc[0], out_g, 8);
  simdgroup_store(acc[1], out_u, 8);
  simdgroup_barrier(mem_flags::mem_threadgroup);
  const int out_half_n = N >> 1;
  const int r_base = sg_row >> 1;
  for (uint i = lane; i < 64u; i += 32u) {
    uint rb = i >> 3;
    uint rn = i & 7u;
    if (r_base + int(rn) < out_half_n) {
      float g = float(static_cast<bfloat16_t>(out_g[i]));
      float u = float(static_cast<bfloat16_t>(out_u[i]));
      float act = (g / (1.0f + metal::exp(-g))) * u;
      y[size_t(rb) * out_half_n + size_t(r_base + int(rn))] = static_cast<bfloat16_t>(act);
    }
  }
"""
    return r"""
  constexpr int BK = 64;
  constexpr int LDW = 64 + 8;
  uint ty = threadgroup_position_in_grid.y;
  uint sg = simdgroup_index_in_threadgroup;
  uint lane = thread_index_in_simdgroup;

  const int sg_row = (int(ty) * NSG + int(sg)) * (8 * RPT);
  if (sg_row >= N) return;
  const int in_w = K / 2;
  const int in_g = K / 64;

  // Ping-pong staging buffer per SG eliminates the second barrier inside the K-loop.
  threadgroup bfloat16_t sw[NSG * 2 * 8 * LDW];
  threadgroup bfloat16_t* sw_sg = sw + sg * (2 * 8 * LDW);

  simdgroup_matrix<float, 8, 8> acc[RPT];
  _Pragma("clang loop unroll(full)")
  for (int rt = 0; rt < RPT; rt++) acc[rt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

  const uint wr_l = lane >> 2;
  const uint wc_l = lane & 3u;
  const float4 inv_s = float4(1.0f, 1.0f / 16.0f, 1.0f / 256.0f, 1.0f / 4096.0f);
  uint step = 0u;

  for (int k = 0; k < K; k += BK) {
    simdgroup_matrix<bfloat16_t, 8, 8> mA[8];
    _Pragma("clang loop unroll(full)")
    for (int kk = 0; kk < 8; kk++) {
      simdgroup_load(mA[kk], x + k + kk * 8, K);
    }

    _Pragma("clang loop unroll(full)")
    for (int rt = 0; rt < RPT; rt++) {
      threadgroup bfloat16_t* sw_cur = sw_sg + (step & 1u) * (8 * LDW);
      step++;
      int g_row = min(sg_row + rt * 8 + int(wr_l), N - 1);
      const device uint8_t* wp = (const device uint8_t*)w + size_t(g_row) * in_w + (k >> 1) + wc_l * 8u;
      uint2 wpk = ((const device uint2*)wp)[0];
      float sc = float(scales[size_t(g_row) * in_g + (k >> 6)]);
      float bi = float(biases[size_t(g_row) * in_g + (k >> 6)]);
      float4 s4 = inv_s * sc;
      ushort u0 = ushort(wpk.x & 0xFFFFu), u1 = ushort(wpk.x >> 16);
      ushort u2 = ushort(wpk.y & 0xFFFFu), u3 = ushort(wpk.y >> 16);
      float4 f0 = float4(float(u0 & 0x000Fu), float(u0 & 0x00F0u), float(u0 & 0x0F00u), float(u0 & 0xF000u)) * s4 + bi;
      float4 f1 = float4(float(u1 & 0x000Fu), float(u1 & 0x00F0u), float(u1 & 0x0F00u), float(u1 & 0xF000u)) * s4 + bi;
      float4 f2 = float4(float(u2 & 0x000Fu), float(u2 & 0x00F0u), float(u2 & 0x0F00u), float(u2 & 0xF000u)) * s4 + bi;
      float4 f3 = float4(float(u3 & 0x000Fu), float(u3 & 0x00F0u), float(u3 & 0x0F00u), float(u3 & 0xF000u)) * s4 + bi;
      threadgroup bfloat16_t* dst = sw_cur + wr_l * LDW + wc_l * 16u;
      ((threadgroup uint2*)(dst +  0u))[0] = as_type<uint2>(bfloat4(bfloat16_t(f0.x), bfloat16_t(f0.y), bfloat16_t(f0.z), bfloat16_t(f0.w)));
      ((threadgroup uint2*)(dst +  4u))[0] = as_type<uint2>(bfloat4(bfloat16_t(f1.x), bfloat16_t(f1.y), bfloat16_t(f1.z), bfloat16_t(f1.w)));
      ((threadgroup uint2*)(dst +  8u))[0] = as_type<uint2>(bfloat4(bfloat16_t(f2.x), bfloat16_t(f2.y), bfloat16_t(f2.z), bfloat16_t(f2.w)));
      ((threadgroup uint2*)(dst + 12u))[0] = as_type<uint2>(bfloat4(bfloat16_t(f3.x), bfloat16_t(f3.y), bfloat16_t(f3.z), bfloat16_t(f3.w)));
      simdgroup_barrier(mem_flags::mem_threadgroup);

      _Pragma("clang loop unroll(full)")
      for (int kk = 0; kk < 8; kk++) {
        simdgroup_matrix<bfloat16_t, 8, 8> mB;
        simdgroup_load(mB, sw_cur + kk * 8, LDW, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(acc[rt], mA[kk], mB, acc[rt]);
      }
    }
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
""" + epi


def _run_qmv4_bf16_mma8(x: mx.array, w: mx.array, scales: mx.array, biases: mx.array, swiglu: bool = False) -> mx.array:
    B, K = int(x.shape[0]), int(x.shape[1])
    N = int(w.shape[0])
    assert B == 8 and K % 64 == 0 and x.dtype == mx.bfloat16 and scales.dtype == mx.bfloat16 and biases.dtype == mx.bfloat16
    assert (not swiglu) or (N % 16 == 0)
    kfn = _qmv4_bf16_mma8_k.get(swiglu)
    if kfn is None:
        kfn = _qmv4_bf16_mma8_k[swiglu] = mx.fast.metal_kernel(
            name="rows_qmv4_bf16_mma8_sw" if swiglu else "rows_qmv4_bf16_mma8",
            input_names=["x", "w", "scales", "biases"],
            output_names=["y"],
            source=_qmv4_bf16_mma8_src(swiglu=swiglu),
            ensure_row_contiguous=True,
        )
    nsg_env = os.environ.get("LOOPLING_DFLASH2_MMA_NSG")
    if nsg_env:
        nsg = int(nsg_env)
        rpt = 2 if swiglu else 1
    elif swiglu:
        nsg, rpt = 1, 2
    else:
        nsg, rpt = 1, 1
    bn = nsg * rpt * 8
    tiles = (N + bn - 1) // bn
    out_n = (N >> 1) if swiglu else N
    return kfn(
        inputs=[x, w, scales, biases],
        template=[("K", K), ("N", N), ("NSG", nsg), ("RPT", rpt)],
        grid=(32, nsg * tiles, 1),
        threadgroup=(32, nsg, 1),
        output_shapes=[(8, out_n)],
        output_dtypes=[mx.bfloat16],
    )[0]


def qmv4_bf16_mma8_rows(x: mx.array, w: mx.array, scales: mx.array, biases: mx.array) -> mx.array:
    """Barrier-free simdgroup_matrix 4-bit bf16 matmul for B==8 (~96% DRAM BW)."""
    return _run_qmv4_bf16_mma8(x, w, scales, biases, swiglu=False)


def qmv4_bf16_mma8_swiglu_rows(x: mx.array, w: mx.array, scales: mx.array, biases: mx.array) -> mx.array:
    """Barrier-free simdgroup_matrix 4-bit bf16 block-8 interleaved gate/up + SwiGLU for B==8."""
    return _run_qmv4_bf16_mma8(x, w, scales, biases, swiglu=True)


_DYN_CONV_S0_BF16_SRC = r"""
  uint tid = thread_position_in_grid.x;
  uint b = thread_position_in_grid.y;
  const uint c = tid * 4u;
  if (c >= HH) return;
  const uint g = c >> 4;
  const uint d_off = b * (4u * GG);
  float d0 = float(dyn[d_off + g]);
  float d1 = float(dyn[d_off + GG + g]);
  float4 b0 = float4(as_type<bfloat4>(((const device uint2*)(base + c))[0]));
  float4 b1 = float4(as_type<bfloat4>(((const device uint2*)(base + HH + c))[0]));
  float4 xc = float4(as_type<bfloat4>(((const device uint2*)(x + size_t(b) * HH + c))[0]));
  float4 xp = (b > 0u) ? float4(as_type<bfloat4>(((const device uint2*)(x + size_t(b - 1u) * HH + c))[0])) : float4(0.0f);
  float4 val = xc * (b0 + d0) + xp * (b1 + d1);
  bfloat4 bv = bfloat4(bfloat16_t(val.x), bfloat16_t(val.y), bfloat16_t(val.z), bfloat16_t(val.w));
  ((device uint2*)(out + size_t(b) * HH + c))[0] = as_type<uint2>(bv);
"""


def dyn_conv_stage0_bf16(x: mx.array, dyn: mx.array, base: mx.array) -> mx.array:
    """Fused BF16 2-tap dynamic convolution (stage 0, no residual): x (B, H), dyn (B, 4*G), base (2, 2, H) -> (B, H) bf16."""
    global _dyn_conv_s0_bf16_k
    BB, HH = int(x.shape[0]), int(x.shape[1])
    GG = HH // 16
    assert HH % 16 == 0 and x.dtype == mx.bfloat16 and dyn.dtype == mx.bfloat16 and base.dtype == mx.bfloat16
    if _dyn_conv_s0_bf16_k is None:
        _dyn_conv_s0_bf16_k = mx.fast.metal_kernel(
            name="dflash2_dyn_conv_s0_bf16",
            input_names=["x", "dyn", "base"],
            output_names=["out"],
            source=_DYN_CONV_S0_BF16_SRC,
            ensure_row_contiguous=True,
        )
    n4 = HH // 4
    tg = 288 if n4 % 288 == 0 else min(256, n4)
    gx = ((n4 + tg - 1) // tg) * tg
    return _dyn_conv_s0_bf16_k(
        inputs=[x, dyn.reshape(BB, 4 * GG), base.reshape(-1)],
        template=[("HH", HH), ("GG", GG)],
        grid=(gx, BB, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(BB, HH)],
        output_dtypes=[mx.bfloat16],
    )[0]


_DYN_CONV_RES_RMS_BF16_SRC = r"""
  uint tid = thread_position_in_threadgroup.x;
  uint b = threadgroup_position_in_grid.y;
  uint sg = tid >> 5;
  uint lane = tid & 31u;

  const uint c0 = tid * 4u;
  const uint c1 = (tid + 288u) * 4u;
  const uint g0 = c0 >> 4;
  const uint g1 = c1 >> 4;
  const uint d_off = b * 576u + 288u;

  float d0_0 = float(dyn[d_off + g0]);
  float d1_0 = float(dyn[d_off + 144u + g0]);
  float d0_1 = float(dyn[d_off + g1]);
  float d1_1 = float(dyn[d_off + 144u + g1]);

  float4 b0_0 = float4(as_type<bfloat4>(((const device uint2*)(base + 4608u + c0))[0]));
  float4 b1_0 = float4(as_type<bfloat4>(((const device uint2*)(base + 6912u + c0))[0]));
  float4 b0_1 = float4(as_type<bfloat4>(((const device uint2*)(base + 4608u + c1))[0]));
  float4 b1_1 = float4(as_type<bfloat4>(((const device uint2*)(base + 6912u + c1))[0]));

  size_t row_off = size_t(b) * 2304u;
  float4 xc0 = float4(as_type<bfloat4>(((const device uint2*)(x + row_off + c0))[0]));
  float4 xc1 = float4(as_type<bfloat4>(((const device uint2*)(x + row_off + c1))[0]));
  float4 xp0 = (b > 0u) ? float4(as_type<bfloat4>(((const device uint2*)(x + size_t(b - 1u) * 2304u + c0))[0])) : float4(0.0f);
  float4 xp1 = (b > 0u) ? float4(as_type<bfloat4>(((const device uint2*)(x + size_t(b - 1u) * 2304u + c1))[0])) : float4(0.0f);
  float4 rs0 = float4(as_type<bfloat4>(((const device uint2*)(residual + row_off + c0))[0]));
  float4 rs1 = float4(as_type<bfloat4>(((const device uint2*)(residual + row_off + c1))[0]));

  float4 v0 = xc0 * (b0_0 + d0_0) + xp0 * (b1_0 + d1_0) + rs0;
  float4 v1 = xc1 * (b0_1 + d0_1) + xp1 * (b1_1 + d1_1) + rs1;
  bfloat4 r0 = bfloat4(bfloat16_t(v0.x), bfloat16_t(v0.y), bfloat16_t(v0.z), bfloat16_t(v0.w));
  bfloat4 r1 = bfloat4(bfloat16_t(v1.x), bfloat16_t(v1.y), bfloat16_t(v1.z), bfloat16_t(v1.w));
  ((device uint2*)(res + row_off + c0))[0] = as_type<uint2>(r0);
  ((device uint2*)(res + row_off + c1))[0] = as_type<uint2>(r1);

  float4 fr0 = float4(r0);
  float4 fr1 = float4(r1);
  float ss = simd_sum(dot(fr0, fr0) + dot(fr1, fr1));
  threadgroup float part[9];
  if (lane == 0u) part[sg] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float tot = simd_sum((lane < 9u) ? part[lane] : 0.0f);
  float inv = metal::rsqrt(tot / 2304.0f + 1e-6f);

  float4 lw0 = float4(as_type<bfloat4>(((const device uint2*)(ln_w + c0))[0]));
  float4 lw1 = float4(as_type<bfloat4>(((const device uint2*)(ln_w + c1))[0]));
  float4 n0 = fr0 * inv * lw0;
  float4 n1 = fr1 * inv * lw1;
  bfloat4 bn0 = bfloat4(bfloat16_t(n0.x), bfloat16_t(n0.y), bfloat16_t(n0.z), bfloat16_t(n0.w));
  bfloat4 bn1 = bfloat4(bfloat16_t(n1.x), bfloat16_t(n1.y), bfloat16_t(n1.z), bfloat16_t(n1.w));
  ((device uint2*)(normed + row_off + c0))[0] = as_type<uint2>(bn0);
  ((device uint2*)(normed + row_off + c1))[0] = as_type<uint2>(bn1);
"""


def dyn_conv_res_rms_bf16(
    x: mx.array, dyn: mx.array, base: mx.array, residual: mx.array, ln_w: mx.array
) -> tuple[mx.array, mx.array]:
    """Fused BF16 stage-1 dynamic conv + residual add + RMSNorm for H=2304 -> (res, normed)."""
    global _dyn_conv_res_rms_bf16_k
    BB, HH = int(x.shape[0]), int(x.shape[1])
    assert HH == 2304 and x.dtype == mx.bfloat16 and residual.dtype == mx.bfloat16 and ln_w.dtype == mx.bfloat16
    if _dyn_conv_res_rms_bf16_k is None:
        _dyn_conv_res_rms_bf16_k = mx.fast.metal_kernel(
            name="dflash2_dyn_conv_res_rms_bf16",
            input_names=["x", "dyn", "base", "residual", "ln_w"],
            output_names=["res", "normed"],
            source=_DYN_CONV_RES_RMS_BF16_SRC,
            ensure_row_contiguous=True,
        )
    return _dyn_conv_res_rms_bf16_k(
        inputs=[x, dyn.reshape(BB, 576), base.reshape(-1), residual, ln_w],
        grid=(288, BB, 1),
        threadgroup=(288, 1, 1),
        output_shapes=[(BB, 2304), (BB, 2304)],
        output_dtypes=[mx.bfloat16, mx.bfloat16],
    )


def _qkv_norm_rope_bf16_src(theta_lit):
    return f"""
  uint bh = threadgroup_position_in_grid.x;
  uint lane = thread_index_in_simdgroup;
  const int tot_h = NH + 2 * NKV;
  const int b = int(bh) / tot_h;
  const int h = int(bh) % tot_h;
  const device bfloat16_t* xp = qkv + size_t(bh) * 128;
  const int i0 = int(lane) * 2;
  const int j0 = i0 + 64;

  uint32_t xlo_u = *(const device uint32_t*)(xp + i0);
  uint32_t xhi_u = *(const device uint32_t*)(xp + j0);

  if (h >= NH + NKV) {{
    int vh = h - (NH + NKV);
    device bfloat16_t* vp = v + (size_t(vh) * BB + b) * 128;
    *(device uint32_t*)(vp + i0) = xlo_u;
    *(device uint32_t*)(vp + j0) = xhi_u;
    return;
  }}

  float2 flo = float2(as_type<bfloat2>(xlo_u));
  float2 fhi = float2(as_type<bfloat2>(xhi_u));
  float ss = simd_sum(flo.x * flo.x + flo.y * flo.y + fhi.x * fhi.x + fhi.y * fhi.y);
  float inv_rms = metal::rsqrt(ss / 128.0f + 1e-6f);
  const device bfloat16_t* nw = (h < NH) ? q_norm : k_norm;
  float2 wlo = float2(as_type<bfloat2>(*(const device uint32_t*)(nw + i0)));
  float2 whi = float2(as_type<bfloat2>(*(const device uint32_t*)(nw + j0)));
  float2 rlo = float2(bfloat2(bfloat16_t(flo.x * inv_rms * wlo.x), bfloat16_t(flo.y * inv_rms * wlo.y)));
  float2 rhi = float2(bfloat2(bfloat16_t(fhi.x * inv_rms * whi.x), bfloat16_t(fhi.y * inv_rms * whi.y)));

  const float p = float(pos_arr[0] + b);
  const float inv_half_d = 1.0f / 64.0f;
  const float theta = {theta_lit};
  float a0 = p * metal::pow(theta, -float(i0) * inv_half_d);
  float a1 = p * metal::pow(theta, -float(i0 + 1) * inv_half_d);
  float c0 = metal::cos(a0), s0 = metal::sin(a0);
  float c1 = metal::cos(a1), s1 = metal::sin(a1);

  bfloat2 olo = bfloat2(bfloat16_t(rlo.x * c0 - rhi.x * s0), bfloat16_t(rlo.y * c1 - rhi.y * s1));
  bfloat2 ohi = bfloat2(bfloat16_t(rhi.x * c0 + rlo.x * s0), bfloat16_t(rhi.y * c1 + rlo.y * s1));
  device bfloat16_t* dst = (h < NH) ? (q + (size_t(h) * BB + b) * 128) : (k + (size_t(h - NH) * BB + b) * 128);
  *(device uint32_t*)(dst + i0) = as_type<uint32_t>(olo);
  *(device uint32_t*)(dst + j0) = as_type<uint32_t>(ohi);
"""


def qkv_norm_rope_bf16(qkv, q_norm, k_norm, pos_arr, nh, nkv, hd, theta):
    """Fused bf16 Q/K RMSNorm + NeoX RoPE + V transpose for (B, (NH+2*NKV)*hd) -> q, k, v."""
    assert hd == 128 and qkv.ndim == 2 and qkv.dtype == mx.bfloat16
    BB = int(qkv.shape[0])
    NH, NKV = int(nh), int(nkv)
    assert qkv.shape[1] == (NH + 2 * NKV) * 128
    th_f = float(theta)
    kfn = _qkv_norm_rope_bf16_k.get(th_f)
    if kfn is None:
        th_lit = flit(th_f)
        sfx = th_lit.replace(".", "p").replace("-", "m").replace("+", "P")
        kfn = _qkv_norm_rope_bf16_k[th_f] = mx.fast.metal_kernel(
            name=f"qkv_norm_rope_bf16_{sfx}",
            input_names=["qkv", "q_norm", "k_norm", "pos_arr"],
            output_names=["q", "k", "v"],
            source=_qkv_norm_rope_bf16_src(th_lit),
            ensure_row_contiguous=True,
        )
    return kfn(
        inputs=[qkv, q_norm, k_norm, pos_arr.reshape(-1)],
        template=[("BB", BB), ("NH", NH), ("NKV", NKV)],
        grid=(BB * (NH + 2 * NKV) * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(1, NH, BB, 128), (1, NKV, BB, 128), (1, NKV, BB, 128)],
        output_dtypes=[mx.bfloat16, mx.bfloat16, mx.bfloat16],
    )


_SDPA_DRAFT_BF16_SRC = r"""
  // Processes 2 query slots (b0, b1) per TG when BB % 2 == 0 to halve KV cache DRAM reads!
  uint bh = threadgroup_position_in_grid.x;
  uint sg = simdgroup_index_in_threadgroup;
  uint lane = thread_index_in_simdgroup;
  int bg = int(bh) / HQ;
  int h = int(bh) % HQ;
  int kvh = h / GQA;
  int b0 = bg * 2;
  int b1 = b0 + 1;

  const int P = dr_pos[0];
  const int s = (WIN > 0) ? max(0, P - WIN) : 0;
  const int n_ctx = max(0, P - s);
  const int tot = n_ctx + BB;

  float4 qv0 = float4(as_type<bfloat4>(((const device uint2*)(q + (size_t(h) * BB + b0) * D + lane * 4))[0])) * scale[0];
  float4 qv1 = float4(as_type<bfloat4>(((const device uint2*)(q + (size_t(h) * BB + b1) * D + lane * 4))[0])) * scale[0];
  const device bfloat16_t* ck_h = ck + size_t(kvh) * CAP * D + lane * 4;
  const device bfloat16_t* cv_h = cv + size_t(kvh) * CAP * D + lane * 4;
  const device bfloat16_t* kc_h = k_cur + size_t(kvh) * BB * D + lane * 4;
  const device bfloat16_t* vc_h = v_cur + size_t(kvh) * BB * D + lane * 4;

  float m0 = -INFINITY, l0 = 0.0f;
  float m1 = -INFINITY, l1 = 0.0f;
  float4 ov0 = float4(0.0f);
  float4 ov1 = float4(0.0f);

  for (int idx = int(sg); idx < tot; idx += NS) {
    const device bfloat16_t* kp = (idx < n_ctx) ? (ck_h + size_t(s + idx) * D) : (kc_h + size_t(idx - n_ctx) * D);
    const device bfloat16_t* vp = (idx < n_ctx) ? (cv_h + size_t(s + idx) * D) : (vc_h + size_t(idx - n_ctx) * D);
    float4 kv4 = float4(as_type<bfloat4>(((const device uint2*)kp)[0]));
    float4 vv4 = float4(as_type<bfloat4>(((const device uint2*)vp)[0]));

    float sc0 = simd_sum(dot(qv0, kv4));
    float nm0 = max(m0, sc0);
    float f0 = (m0 > -INFINITY) ? fast::exp(m0 - nm0) : 0.0f;
    float e0 = fast::exp(sc0 - nm0);
    m0 = nm0; l0 = l0 * f0 + e0; ov0 = ov0 * f0 + e0 * vv4;

    float sc1 = simd_sum(dot(qv1, kv4));
    float nm1 = max(m1, sc1);
    float f1 = (m1 > -INFINITY) ? fast::exp(m1 - nm1) : 0.0f;
    float e1 = fast::exp(sc1 - nm1);
    m1 = nm1; l1 = l1 * f1 + e1; ov1 = ov1 * f1 + e1 * vv4;
  }

  threadgroup float tg_m[2][NS];
  threadgroup float tg_l[2][NS];
  threadgroup float4 tg_o[2][NS][32];
  if (lane == 0) {
    tg_m[0][sg] = m0; tg_l[0][sg] = l0;
    tg_m[1][sg] = m1; tg_l[1][sg] = l1;
  }
  tg_o[0][sg][lane] = ov0;
  tg_o[1][sg][lane] = ov1;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sg < 2u) {
    uint qidx = sg;
    int bo = (qidx == 0u) ? b0 : b1;
    float gm = -INFINITY;
    for (int s0 = 0; s0 < NS; s0++) gm = max(gm, tg_m[qidx][s0]);
    float gl = 0.0f;
    float4 go = float4(0.0f);
    for (int s0 = 0; s0 < NS; s0++) {
      float ms0 = tg_m[qidx][s0];
      if (ms0 > -INFINITY) {
        float w = fast::exp(ms0 - gm);
        gl += w * tg_l[qidx][s0];
        go += w * tg_o[qidx][s0][lane];
      }
    }
    go *= (gl > 0.0f) ? (1.0f / gl) : 0.0f;
    bfloat4 bgv = bfloat4(bfloat16_t(go.x), bfloat16_t(go.y), bfloat16_t(go.z), bfloat16_t(go.w));
    ((device uint2*)(out + (size_t(bo) * HQ + h) * D + lane * 4))[0] = as_type<uint2>(bgv);
  }
"""


_SDPA_DRAFT_BF16_Q1_SRC = r"""
  uint bh = threadgroup_position_in_grid.x;
  uint sg = simdgroup_index_in_threadgroup;
  uint lane = thread_index_in_simdgroup;
  int b = int(bh) / HQ;
  int h = int(bh) % HQ;
  int kvh = h / GQA;

  const int P = dr_pos[0];
  const int s = (WIN > 0) ? max(0, P - WIN) : 0;
  const int n_ctx = max(0, P - s);
  const int tot = n_ctx + BB;

  float4 qv = float4(as_type<bfloat4>(((const device uint2*)(q + (size_t(h) * BB + b) * D + lane * 4))[0])) * scale[0];
  const device bfloat16_t* ck_h = ck + size_t(kvh) * CAP * D + lane * 4;
  const device bfloat16_t* cv_h = cv + size_t(kvh) * CAP * D + lane * 4;
  const device bfloat16_t* kc_h = k_cur + size_t(kvh) * BB * D + lane * 4;
  const device bfloat16_t* vc_h = v_cur + size_t(kvh) * BB * D + lane * 4;

  float m_sg = -INFINITY;
  float l_sg = 0.0f;
  float4 ov4 = float4(0.0f);

  for (int idx = int(sg); idx < tot; idx += NS) {
    const device bfloat16_t* kp = (idx < n_ctx) ? (ck_h + size_t(s + idx) * D) : (kc_h + size_t(idx - n_ctx) * D);
    const device bfloat16_t* vp = (idx < n_ctx) ? (cv_h + size_t(s + idx) * D) : (vc_h + size_t(idx - n_ctx) * D);
    float4 kv4 = float4(as_type<bfloat4>(((const device uint2*)kp)[0]));
    float4 vv4 = float4(as_type<bfloat4>(((const device uint2*)vp)[0]));
    float score = simd_sum(dot(qv, kv4));
    float new_max = max(m_sg, score);
    float factor = (m_sg > -INFINITY) ? fast::exp(m_sg - new_max) : 0.0f;
    float exp_s = fast::exp(score - new_max);
    m_sg = new_max;
    l_sg = l_sg * factor + exp_s;
    ov4 = ov4 * factor + exp_s * vv4;
  }

  threadgroup float tg_m[NS];
  threadgroup float tg_l[NS];
  threadgroup float4 tg_o[NS][32];
  if (lane == 0) {
    tg_m[sg] = m_sg;
    tg_l[sg] = l_sg;
  }
  tg_o[sg][lane] = ov4;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sg == 0) {
    float gm = -INFINITY;
    for (int s0 = 0; s0 < NS; s0++) gm = max(gm, tg_m[s0]);
    float gl = 0.0f;
    float4 go = float4(0.0f);
    for (int s0 = 0; s0 < NS; s0++) {
      float ms0 = tg_m[s0];
      if (ms0 > -INFINITY) {
        float w = fast::exp(ms0 - gm);
        gl += w * tg_l[s0];
        go += w * tg_o[s0][lane];
      }
    }
    go *= (gl > 0.0f) ? (1.0f / gl) : 0.0f;
    bfloat4 bgv = bfloat4(bfloat16_t(go.x), bfloat16_t(go.y), bfloat16_t(go.z), bfloat16_t(go.w));
    ((device uint2*)(out + (size_t(b) * HQ + h) * D + lane * 4))[0] = as_type<uint2>(bgv);
  }
"""


def sdpa_draft_bf16_gpu(q, ck, cv, k_cur, v_cur, dr_pos_arr, *, scale, window):
    """Zero-copy bf16 bidirectional block attention -> (B, HQ*D) bf16."""
    global _sdpa_draft_bf16_k, _sdpa_draft_bf16_q1_k
    _, HQ, BB, D = q.shape
    HKV = int(k_cur.shape[1])
    CAP = int(ck.shape[2])
    assert D == 128 and HQ % HKV == 0 and q.dtype == mx.bfloat16 and ck.dtype == mx.bfloat16
    sck = float(scale)
    sc = _SCALE_F32_CACHE.get(sck)
    if sc is None:
        sc = _SCALE_F32_CACHE[sck] = mx.array([sck], mx.float32)
    ns = 8
    use_q2 = (BB % 2 == 0) and os.environ.get("LOOPLING_DFLASH2_SDPA_Q2", "0") not in ("0", "", "false", "False")
    if use_q2:
        if _sdpa_draft_bf16_k is None:
            _sdpa_draft_bf16_k = mx.fast.metal_kernel(
                name="sdpa_draft_bf16_gpu_q2",
                input_names=["q", "ck", "cv", "k_cur", "v_cur", "dr_pos", "scale"],
                output_names=["out"],
                source=_SDPA_DRAFT_BF16_SRC,
                ensure_row_contiguous=True,
            )
        return _sdpa_draft_bf16_k(
            inputs=[q, ck, cv, k_cur, v_cur, dr_pos_arr.reshape(-1), sc],
            template=[("BB", BB), ("HQ", HQ), ("GQA", HQ // HKV), ("D", D), ("CAP", CAP), ("WIN", int(window)), ("NS", ns)],
            grid=((BB // 2) * HQ * 32 * ns, 1, 1),
            threadgroup=(32 * ns, 1, 1),
            output_shapes=[(BB, HQ * D)],
            output_dtypes=[mx.bfloat16],
        )[0]
    if _sdpa_draft_bf16_q1_k is None:
        _sdpa_draft_bf16_q1_k = mx.fast.metal_kernel(
            name="sdpa_draft_bf16_gpu",
            input_names=["q", "ck", "cv", "k_cur", "v_cur", "dr_pos", "scale"],
            output_names=["out"],
            source=_SDPA_DRAFT_BF16_Q1_SRC,
            ensure_row_contiguous=True,
        )
    return _sdpa_draft_bf16_q1_k(
        inputs=[q, ck, cv, k_cur, v_cur, dr_pos_arr.reshape(-1), sc],
        template=[("BB", BB), ("HQ", HQ), ("GQA", HQ // HKV), ("D", D), ("CAP", CAP), ("WIN", int(window)), ("NS", ns)],
        grid=(BB * HQ * 32 * ns, 1, 1),
        threadgroup=(32 * ns, 1, 1),
        output_shapes=[(BB, HQ * D)],
        output_dtypes=[mx.bfloat16],
    )[0]


_QMV4_BF16_TOP16_P1_SRC = r"""
  constexpr int group_size = 64;
  constexpr int NFULL = K / 512;
  constexpr int TAIL = K % 512;
  static_assert(K % 256 == 0, "qmv4 bf16 top16 u2 needs K % 256 == 0");

  uint sp_id = threadgroup_position_in_grid.y;
  uint simd_gid = simdgroup_index_in_threadgroup;
  uint simd_lid = thread_index_in_simdgroup;

  float bv0[BB], bv1[BB];
  uint  bi0[BB], bi1[BB];
  for (int b = 0; b < BB; b++) {
    bv0[b] = -INFINITY; bv1[b] = -INFINITY;
    bi0[b] = 0xFFFFFFFFu; bi1[b] = 0xFFFFFFFFu;
  }

  const int in_w = K / 2;
  const int in_g = K / group_size;
  const float4 inv_s = float4(1.0f, 1.0f / 16.0f, 1.0f / 256.0f, 1.0f / 4096.0f);

  for (int ty = int(sp_id); ty < TOT_TILES; ty += SP) {
    const int out_row = ty * (NSG * RPS) + int(simd_gid) * RPS;
    if (out_row < N) {
      const int last_row = min(RPS - 1, N - 1 - out_row);
      float result[BB][RPS];
      _Pragma("clang loop unroll(full)")
      for (int b = 0; b < BB; b++)
        _Pragma("clang loop unroll(full)")
        for (int r = 0; r < RPS; r++) result[b][r] = 0.0f;

      const device uint8_t* ws = (const device uint8_t*)w + size_t(out_row) * in_w + simd_lid * 8;
      const device bfloat16_t* sc_p = scales + size_t(out_row) * in_g + (simd_lid >> 2);
      const device bfloat16_t* bi_p = biases + size_t(out_row) * in_g + (simd_lid >> 2);
      const device bfloat16_t* xp = x + simd_lid * 16;

      for (int it = 0; it < NFULL; it++) {
        thread uint2 wr[RPS];
        thread float sc[RPS], bi[RPS];
        _Pragma("clang loop unroll(full)")
        for (int row = 0; row < RPS; row++) {
          int src = min(row, last_row);
          wr[row] = ((const device uint2*)(ws + size_t(src) * in_w))[0];
          sc[row] = float(sc_p[size_t(src) * in_g]);
          bi[row] = float(bi_p[size_t(src) * in_g]);
        }
        _Pragma("clang loop unroll(full)")
        for (int b = 0; b < BB; b++) {
          const device bfloat16_t* xb = xp + size_t(b) * K;
          uint4 xpk0 = ((const device uint4*)xb)[0];
          uint4 xpk1 = ((const device uint4*)(xb + 8))[0];
          float4 a0 = float4(as_type<bfloat4>(xpk0.xy));
          float4 a1 = float4(as_type<bfloat4>(xpk0.zw));
          float4 a2 = float4(as_type<bfloat4>(xpk1.xy));
          float4 a3 = float4(as_type<bfloat4>(xpk1.zw));
          float smb = ((a0.x + a0.y + a0.z + a0.w) + (a1.x + a1.y + a1.z + a1.w))
                    + ((a2.x + a2.y + a2.z + a2.w) + (a3.x + a3.y + a3.z + a3.w));
          float4 xf0 = a0 * inv_s, xf1 = a1 * inv_s, xf2 = a2 * inv_s, xf3 = a3 * inv_s;
          _Pragma("clang loop unroll(full)")
          for (int row = 0; row < RPS; row++) {
            uint2 wv = wr[row];
            ushort u0 = ushort(wv.x & 0xFFFFu), u1 = ushort(wv.x >> 16);
            ushort u2 = ushort(wv.y & 0xFFFFu), u3 = ushort(wv.y >> 16);
            float4 w0 = float4(float(u0 & 0x000Fu), float(u0 & 0x00F0u), float(u0 & 0x0F00u), float(u0 & 0xF000u));
            float4 w1 = float4(float(u1 & 0x000Fu), float(u1 & 0x00F0u), float(u1 & 0x0F00u), float(u1 & 0xF000u));
            float4 w2 = float4(float(u2 & 0x000Fu), float(u2 & 0x00F0u), float(u2 & 0x0F00u), float(u2 & 0xF000u));
            float4 w3 = float4(float(u3 & 0x000Fu), float(u3 & 0x00F0u), float(u3 & 0x0F00u), float(u3 & 0xF000u));
            float dp = (dot(xf0, w0) + dot(xf1, w1)) + (dot(xf2, w2) + dot(xf3, w3));
            result[b][row] += sc[row] * dp + smb * bi[row];
          }
        }
        ws += 256;
        sc_p += 8;
        bi_p += 8;
        xp += 512;
      }

      if (TAIL == 256) {
        const device uint8_t* ws_t = (const device uint8_t*)w + size_t(out_row) * in_w + NFULL * 256 + simd_lid * 4;
        const device bfloat16_t* sc_t = scales + size_t(out_row) * in_g + NFULL * 8 + (simd_lid >> 3);
        const device bfloat16_t* bi_t = biases + size_t(out_row) * in_g + NFULL * 8 + (simd_lid >> 3);
        const device bfloat16_t* xp_t = x + NFULL * 512 + simd_lid * 8;
        thread uint32_t wr[RPS];
        thread float sc[RPS], bi[RPS];
        _Pragma("clang loop unroll(full)")
        for (int row = 0; row < RPS; row++) {
          int src = min(row, last_row);
          wr[row] = ((const device uint32_t*)(ws_t + size_t(src) * in_w))[0];
          sc[row] = float(sc_t[size_t(src) * in_g]);
          bi[row] = float(bi_t[size_t(src) * in_g]);
        }
        _Pragma("clang loop unroll(full)")
        for (int b = 0; b < BB; b++) {
          uint4 xpk = ((const device uint4*)(xp_t + size_t(b) * K))[0];
          float4 a0 = float4(as_type<bfloat4>(xpk.xy));
          float4 a1 = float4(as_type<bfloat4>(xpk.zw));
          float smb = (a0.x + a0.y + a0.z + a0.w) + (a1.x + a1.y + a1.z + a1.w);
          float4 xf0 = a0 * inv_s, xf1 = a1 * inv_s;
          _Pragma("clang loop unroll(full)")
          for (int row = 0; row < RPS; row++) {
            uint32_t wv = wr[row];
            ushort u0 = ushort(wv & 0xFFFFu), u1 = ushort(wv >> 16);
            float4 w0 = float4(float(u0 & 0x000Fu), float(u0 & 0x00F0u), float(u0 & 0x0F00u), float(u0 & 0xF000u));
            float4 w1 = float4(float(u1 & 0x000Fu), float(u1 & 0x00F0u), float(u1 & 0x0F00u), float(u1 & 0xF000u));
            result[b][row] += sc[row] * (dot(xf0, w0) + dot(xf1, w1)) + smb * bi[row];
          }
        }
      }

      for (int row = 0; row <= last_row; row++) {
        uint idx = uint(out_row + row);
        for (int b = 0; b < BB; b++) {
          float r = simd_sum(result[b][row]);
          if (simd_lid == 0) {
            float val = float(static_cast<bfloat16_t>(r));
            if (val > bv0[b] || (val == bv0[b] && idx < bi0[b])) {
              bv1[b] = bv0[b]; bi1[b] = bi0[b]; bv0[b] = val; bi0[b] = idx;
            } else if (val > bv1[b] || (val == bv1[b] && idx < bi1[b])) {
              bv1[b] = val; bi1[b] = idx;
            }
          }
        }
      }
    }
  }

  threadgroup float tg_v0[NSG][BB], tg_v1[NSG][BB];
  threadgroup uint  tg_i0[NSG][BB], tg_i1[NSG][BB];
  if (simd_lid == 0) {
    for (int b = 0; b < BB; b++) {
      tg_v0[simd_gid][b] = bv0[b]; tg_i0[simd_gid][b] = bi0[b];
      tg_v1[simd_gid][b] = bv1[b]; tg_i1[simd_gid][b] = bi1[b];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_lid == 0) {
    for (int b = int(simd_gid); b < BB; b += NSG) {
      float fv[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
      uint  fi[4] = {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu};
      for (int s = 0; s < NSG; s++) {
        float vs[2] = {tg_v0[s][b], tg_v1[s][b]};
        uint  is[2] = {tg_i0[s][b], tg_i1[s][b]};
        for (int t = 0; t < 2; t++) {
          float v = vs[t]; uint id = is[t];
          if (v > fv[3] || (v == fv[3] && id < fi[3])) {
            int pos = 3;
            while (pos > 0 && (v > fv[pos - 1] || (v == fv[pos - 1] && id < fi[pos - 1]))) {
              fv[pos] = fv[pos - 1]; fi[pos] = fi[pos - 1]; pos--;
            }
            fv[pos] = v; fi[pos] = id;
          }
        }
      }
      size_t o = size_t(b) * (4 * SP) + size_t(sp_id) * 4;
      for (int k = 0; k < 4; k++) {
        cand_val[o + k] = fv[k];
        cand_idx[o + k] = fi[k];
      }
    }
  }
"""

_TOP16_REDUCE_P2_SRC = r"""
  uint tid = thread_position_in_threadgroup.x;
  uint b = threadgroup_position_in_grid.y;
  uint sg = tid >> 5;
  uint lane = tid & 31u;

  const device float* cv = cand_val + size_t(b) * MM;
  const device uint*  ci = cand_idx + size_t(b) * MM;

  float v[IT];
  uint  id[IT];
  for (int j = 0; j < IT; j++) {
    int idx_m = int(tid) + j * 256;
    if (idx_m < MM) {
      v[j] = cv[idx_m];
      id[j] = ci[idx_m];
    } else {
      v[j] = -INFINITY;
      id[j] = 0xFFFFFFFFu;
    }
  }

  threadgroup float sg_v[8];
  threadgroup uint  sg_i[8];
  threadgroup uint  win_i;

  for (int k = 0; k < 16; k++) {
    float lv = -INFINITY;
    uint  li = 0xFFFFFFFFu;
    for (int j = 0; j < IT; j++) {
      if (v[j] > lv || (v[j] == lv && id[j] < li)) {
        lv = v[j];
        li = id[j];
      }
    }
    for (uint d = 16u; d > 0u; d >>= 1) {
      float ov = simd_shuffle_down(lv, d);
      uint  oi = simd_shuffle_down(li, d);
      if (ov > lv || (ov == lv && oi < li)) {
        lv = ov;
        li = oi;
      }
    }
    if (lane == 0u) {
      sg_v[sg] = lv;
      sg_i[sg] = li;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0u) {
      float gv = (lane < 8u) ? sg_v[lane] : -INFINITY;
      uint  gi = (lane < 8u) ? sg_i[lane] : 0xFFFFFFFFu;
      for (uint d = 4u; d > 0u; d >>= 1) {
        float ov = simd_shuffle_down(gv, d);
        uint  oi = simd_shuffle_down(gi, d);
        if (ov > gv || (ov == gv && oi < gi)) {
          gv = ov;
          gi = oi;
        }
      }
      if (lane == 0u) {
        win_i = gi;
        out_val[size_t(b) * 16u + uint(k)] = gv;
        out_idx[size_t(b) * 16u + uint(k)] = (gi == 0xFFFFFFFFu) ? 0u : gi;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint wi = win_i;
    if (wi != 0xFFFFFFFFu) {
      for (int j = 0; j < IT; j++) {
        if (id[j] == wi) v[j] = -INFINITY;
      }
    }
  }
"""


_QMV4_BF16_TOP16_MMA_P1_SRC = r"""
  constexpr int BK = 64;
  constexpr int LDW = 64 + 8;
  uint sp_id = threadgroup_position_in_grid.y;
  uint sg = simdgroup_index_in_threadgroup;
  uint lane = thread_index_in_simdgroup;

  const int in_w = K / 2;
  const int in_g = K / 64;
  const uint wr_l = lane >> 2;
  const uint wc_l = lane & 3u;
  const float4 inv_s = float4(1.0f, 1.0f / 16.0f, 1.0f / 256.0f, 1.0f / 4096.0f);

  threadgroup bfloat16_t sw[NSG * 2 * 8 * LDW];
  threadgroup bfloat16_t* sw_sg = sw + sg * (2 * 8 * LDW);

  float bv0[BB], bv1[BB];
  uint  bi0[BB], bi1[BB];
  for (int b = 0; b < BB; b++) {
    bv0[b] = -INFINITY; bv1[b] = -INFINITY;
    bi0[b] = 0xFFFFFFFFu; bi1[b] = 0xFFFFFFFFu;
  }

  for (int ty = int(sp_id); ty < TOT_TILES; ty += SP) {
    const int sg_row = (ty * NSG + int(sg)) * (8 * RPT);
    if (sg_row >= N) continue;

    simdgroup_matrix<float, 8, 8> acc[RPT];
    _Pragma("clang loop unroll(full)")
    for (int rt = 0; rt < RPT; rt++) acc[rt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    uint step = 0u;
    for (int k = 0; k < K; k += BK) {
      simdgroup_matrix<bfloat16_t, 8, 8> mA[8];
      _Pragma("clang loop unroll(full)")
      for (int kk = 0; kk < 8; kk++) {
        simdgroup_load(mA[kk], x + k + kk * 8, K);
      }
      _Pragma("clang loop unroll(full)")
      for (int rt = 0; rt < RPT; rt++) {
        threadgroup bfloat16_t* sw_cur = sw_sg + (step & 1u) * (8 * LDW);
        step++;
        int g_row = min(sg_row + rt * 8 + int(wr_l), N - 1);
        const device uint8_t* wp = (const device uint8_t*)w + size_t(g_row) * in_w + (k >> 1) + wc_l * 8u;
        uint2 wpk = ((const device uint2*)wp)[0];
        float sc = float(scales[size_t(g_row) * in_g + (k >> 6)]);
        float bi = float(biases[size_t(g_row) * in_g + (k >> 6)]);
        float4 s4 = inv_s * sc;
        ushort u0 = ushort(wpk.x & 0xFFFFu), u1 = ushort(wpk.x >> 16);
        ushort u2 = ushort(wpk.y & 0xFFFFu), u3 = ushort(wpk.y >> 16);
        float4 f0 = float4(float(u0 & 0x000Fu), float(u0 & 0x00F0u), float(u0 & 0x0F00u), float(u0 & 0xF000u)) * s4 + bi;
        float4 f1 = float4(float(u1 & 0x000Fu), float(u1 & 0x00F0u), float(u1 & 0x0F00u), float(u1 & 0xF000u)) * s4 + bi;
        float4 f2 = float4(float(u2 & 0x000Fu), float(u2 & 0x00F0u), float(u2 & 0x0F00u), float(u2 & 0xF000u)) * s4 + bi;
        float4 f3 = float4(float(u3 & 0x000Fu), float(u3 & 0x00F0u), float(u3 & 0x0F00u), float(u3 & 0xF000u)) * s4 + bi;
        threadgroup bfloat16_t* dst = sw_cur + wr_l * LDW + wc_l * 16u;
        ((threadgroup uint2*)(dst +  0u))[0] = as_type<uint2>(bfloat4(bfloat16_t(f0.x), bfloat16_t(f0.y), bfloat16_t(f0.z), bfloat16_t(f0.w)));
        ((threadgroup uint2*)(dst +  4u))[0] = as_type<uint2>(bfloat4(bfloat16_t(f1.x), bfloat16_t(f1.y), bfloat16_t(f1.z), bfloat16_t(f1.w)));
        ((threadgroup uint2*)(dst +  8u))[0] = as_type<uint2>(bfloat4(bfloat16_t(f2.x), bfloat16_t(f2.y), bfloat16_t(f2.z), bfloat16_t(f2.w)));
        ((threadgroup uint2*)(dst + 12u))[0] = as_type<uint2>(bfloat4(bfloat16_t(f3.x), bfloat16_t(f3.y), bfloat16_t(f3.z), bfloat16_t(f3.w)));
        simdgroup_barrier(mem_flags::mem_threadgroup);
        _Pragma("clang loop unroll(full)")
        for (int kk = 0; kk < 8; kk++) {
          simdgroup_matrix<bfloat16_t, 8, 8> mB;
          simdgroup_load(mB, sw_cur + kk * 8, LDW, ulong2(0, 0), true);
          simdgroup_multiply_accumulate(acc[rt], mA[kk], mB, acc[rt]);
        }
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float* out_s = (threadgroup float*)sw_sg;
    _Pragma("clang loop unroll(full)")
    for (int rt = 0; rt < RPT; rt++) {
      simdgroup_store(acc[rt], out_s, 8);
      simdgroup_barrier(mem_flags::mem_threadgroup);
      if (lane == 0) {
        int r0 = sg_row + rt * 8;
        int n_valid = min(8, N - r0);
        for (int b = 0; b < BB; b++) {
          for (int r = 0; r < n_valid; r++) {
            float val = float(static_cast<bfloat16_t>(out_s[b * 8 + r]));
            uint idx = uint(r0 + r);
            if (val > bv0[b] || (val == bv0[b] && idx < bi0[b])) {
              bv1[b] = bv0[b]; bi1[b] = bi0[b]; bv0[b] = val; bi0[b] = idx;
            } else if (val > bv1[b] || (val == bv1[b] && idx < bi1[b])) {
              bv1[b] = val; bi1[b] = idx;
            }
          }
        }
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  threadgroup float tg_v0[NSG][BB], tg_v1[NSG][BB];
  threadgroup uint  tg_i0[NSG][BB], tg_i1[NSG][BB];
  if (lane == 0) {
    for (int b = 0; b < BB; b++) {
      tg_v0[sg][b] = bv0[b]; tg_i0[sg][b] = bi0[b];
      tg_v1[sg][b] = bv1[b]; tg_i1[sg][b] = bi1[b];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) {
    for (int b = int(sg); b < BB; b += NSG) {
      float fv[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
      uint  fi[4] = {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu};
      for (int s = 0; s < NSG; s++) {
        float vs[2] = {tg_v0[s][b], tg_v1[s][b]};
        uint  is[2] = {tg_i0[s][b], tg_i1[s][b]};
        for (int t = 0; t < 2; t++) {
          float v = vs[t]; uint id = is[t];
          if (v > fv[3] || (v == fv[3] && id < fi[3])) {
            int pos = 3;
            while (pos > 0 && (v > fv[pos - 1] || (v == fv[pos - 1] && id < fi[pos - 1]))) {
              fv[pos] = fv[pos - 1]; fi[pos] = fi[pos - 1]; pos--;
            }
            fv[pos] = v; fi[pos] = id;
          }
        }
      }
      size_t o = size_t(b) * (4 * SP) + size_t(sp_id) * 4;
      for (int k = 0; k < 4; k++) { cand_val[o + k] = fv[k]; cand_idx[o + k] = fi[k]; }
    }
  }
"""


def qmv4_bf16_top16_rows(x: mx.array, w: mx.array, scales: mx.array, biases: mx.array, sp: int = 512):
    """Fused 4-bit BF16 LM head + Top-16 extraction (MMA or U2) -> (top_idx (B, 16) i32, top_val (B, 16) f32)."""
    global _qmv4_bf16_top16_p2_k
    BB, K = int(x.shape[0]), int(x.shape[1])
    N = int(w.shape[0])
    assert 1 <= BB <= 8 and K % 256 == 0 and x.dtype == mx.bfloat16
    assert w.shape[1] * 8 == K and scales.dtype == mx.bfloat16 and biases.dtype == mx.bfloat16
    mode = os.environ.get("LOOPLING_DFLASH2_TOP16_MODE", "mma")
    if _qmv4_bf16_top16_p2_k is None:
        _qmv4_bf16_top16_p2_k = mx.fast.metal_kernel(
            name="qmv4_bf16_top16_p2",
            input_names=["cand_val", "cand_idx"],
            output_names=["out_val", "out_idx"],
            source=_TOP16_REDUCE_P2_SRC,
            ensure_row_contiguous=True,
        )

    if mode == "mma":
        cfg_env = os.environ.get("LOOPLING_DFLASH2_TOP16_MMA_CFG")
        if cfg_env:
            nsg, rpt, sp_i = (int(t) for t in cfg_env.split(","))
        else:
            nsg, rpt = 2, 2
            sp_i = int(os.environ.get("LOOPLING_DFLASH2_TOP16_SP", "1024"))
        bn = 8 * nsg * rpt
        tot_tiles = (N + bn - 1) // bn
        sp_i = max(1, min(sp_i, tot_tiles))
        kp1 = _qmv4_bf16_top16_mma_p1_k.get(BB)
        if kp1 is None:
            kp1 = _qmv4_bf16_top16_mma_p1_k[BB] = mx.fast.metal_kernel(
                name=f"qmv4_bf16_top16_mma_p1_b{BB}",
                input_names=["x", "w", "scales", "biases"],
                output_names=["cand_val", "cand_idx"],
                source=_QMV4_BF16_TOP16_MMA_P1_SRC,
                ensure_row_contiguous=True,
            )
        if BB < 8:
            zkey = (8 - BB, K)
            zpad = _zero_pad_bf16_cache.get(zkey)
            if zpad is None:
                zpad = _zero_pad_bf16_cache[zkey] = mx.zeros(zkey, mx.bfloat16)
            x8 = mx.concatenate([x, zpad], axis=0)
        else:
            x8 = x
        mm = 4 * sp_i
        cand_val, cand_idx = kp1(
            inputs=[x8, w, scales, biases],
            template=[("BB", BB), ("K", K), ("N", N), ("SP", sp_i), ("TOT_TILES", tot_tiles), ("NSG", nsg), ("RPT", rpt)],
            grid=(32, nsg * sp_i, 1),
            threadgroup=(32, nsg, 1),
            output_shapes=[(BB, mm), (BB, mm)],
            output_dtypes=[mx.float32, mx.uint32],
        )
    else:
        cfg_env = os.environ.get("LOOPLING_DFLASH2_TOP16_CFG")
        if cfg_env:
            nsg, rps, sp_i = (int(t) for t in cfg_env.split(","))
        else:
            nsg, rps = 4, 2
            sp_i = int(os.environ.get("LOOPLING_DFLASH2_TOP16_SP", str(sp)))
        bn = nsg * rps
        tot_tiles = (N + bn - 1) // bn
        sp_i = max(1, min(sp_i, tot_tiles))
        kp1 = _qmv4_bf16_top16_p1_k.get(BB)
        if kp1 is None:
            kp1 = _qmv4_bf16_top16_p1_k[BB] = mx.fast.metal_kernel(
                name=f"qmv4_bf16_top16_u2_p1_b{BB}",
                input_names=["x", "w", "scales", "biases"],
                output_names=["cand_val", "cand_idx"],
                source=_QMV4_BF16_TOP16_P1_SRC,
                ensure_row_contiguous=True,
            )
        mm = 4 * sp_i
        cand_val, cand_idx = kp1(
            inputs=[x, w, scales, biases],
            template=[("BB", BB), ("K", K), ("N", N), ("SP", sp_i), ("TOT_TILES", tot_tiles), ("NSG", nsg), ("RPS", rps)],
            grid=(32, nsg * sp_i, 1),
            threadgroup=(32, nsg, 1),
            output_shapes=[(BB, mm), (BB, mm)],
            output_dtypes=[mx.float32, mx.uint32],
        )

    it = (mm + 255) // 256
    out_val, out_idx = _qmv4_bf16_top16_p2_k(
        inputs=[cand_val, cand_idx],
        template=[("MM", mm), ("IT", it)],
        grid=(256, BB, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(BB, 16), (BB, 16)],
        output_dtypes=[mx.float32, mx.int32],
    )
    return out_idx, out_val


_DFLASH2_PACK_CHAIN_SRC = r"""
  if (thread_position_in_grid.x != 0u) return;
  uint anc_tok = anchor[0];
  block_toks[0] = anc_tok;
  parents_out[0] = -1;
  depths_out[0] = 0;

  int slot = 0;
  float best = sc0[0];
  for (int j = 1; j < 16; j++) {
    float v = sc0[j];
    if (v > best) { best = v; slot = j; }
  }
  block_toks[1] = uint(top_idx[slot]);
  parents_out[1] = 0;
  depths_out[1] = 1;

  for (int p = 1; p < WW - 1; p++) {
    const device float* row = sc16 + ((p - 1) * 16 + slot) * 16;
    int ns = 0;
    float nb = row[0];
    for (int j = 1; j < 16; j++) {
      float v = row[j];
      if (v > nb) { nb = v; ns = j; }
    }
    slot = ns;
    block_toks[p + 1] = uint(top_idx[p * 16 + slot]);
    parents_out[p + 1] = p;
    depths_out[p + 1] = p + 1;
  }

  for (int b = 0; b < WW; b++) {
    conv_anc[b * 3 + 0] = (b >= 3) ? (b - 3) : -(b + 1);
    conv_anc[b * 3 + 1] = (b >= 2) ? (b - 2) : -(b + 2);
    conv_anc[b * 3 + 2] = (b >= 1) ? (b - 1) : -3;
    tree_meta[b * 9 + 0] = b;
    for (int k = 0; k < 8; k++) {
      tree_meta[b * 9 + 1 + k] = (k <= b) ? k : 0;
    }
  }
"""


def dflash2_pack_chain_gpu(top_idx: mx.array, sc0: mx.array, sc16: mx.array, anchor: mx.array, W: int):
    """Zero-sync greedy selector chain packer on GPU -> (block_toks, parents, depths, conv_anc, tree_meta)."""
    global _dflash2_pack_chain_k
    WW = int(W)
    assert 2 <= WW <= 8
    if _dflash2_pack_chain_k is None:
        _dflash2_pack_chain_k = mx.fast.metal_kernel(
            name="dflash2_pack_chain_gpu",
            input_names=["top_idx", "sc0", "sc16", "anchor"],
            output_names=["block_toks", "parents_out", "depths_out", "conv_anc", "tree_meta"],
            source=_DFLASH2_PACK_CHAIN_SRC,
            ensure_row_contiguous=True,
        )
    anc_u = anchor.reshape(-1)
    if anc_u.dtype != mx.uint32:
        anc_u = anc_u.astype(mx.uint32)
    return _dflash2_pack_chain_k(
        inputs=[top_idx.reshape(-1), sc0.reshape(-1), sc16.reshape(-1), anc_u],
        template=[("WW", WW)],
        grid=(32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(WW,), (WW,), (WW,), (WW, 3), (WW, 9)],
        output_dtypes=[mx.uint32, mx.int32, mx.int32, mx.int32, mx.int32],
    )


_DFLASH2_PACK_TREE_DINK_SRC = r"""
  if (thread_position_in_grid.x != 0u) return;

  const float slope = prior_cfg[0];
  float intercepts[7];
  for (int i = 0; i < 7; i++) intercepts[i] = prior_cfg[1 + i];

  // Candidate pool (at most 4 + 7*3 = 25 candidates -> 32 slots plenty)
  constexpr int MAX_C = 32;
  float c_jp[MAX_C];
  int   c_dep[MAX_C];
  int   c_par[MAX_C];
  int   c_slot[MAX_C];
  uint  c_tok[MAX_C];
  bool  c_used[MAX_C];
  int   n_cand = 0;

  // Softmax helper over 16 logits + top-K push
  float p0[16];
  float mx0 = sc0[0];
  for (int j = 1; j < 16; j++) mx0 = max(mx0, sc0[j]);
  float sm0 = 0.0f;
  for (int j = 0; j < 16; j++) { p0[j] = metal::exp(sc0[j] - mx0); sm0 += p0[j]; }
  float inv0 = 1.0f / max(sm0, 1e-12f);
  for (int j = 0; j < 16; j++) p0[j] *= inv0;

  // Push top fan0 = min(4, WW - 1) from p0
  int fan0 = min(4, WW - 1);
  bool picked0[16];
  for (int j = 0; j < 16; j++) picked0[j] = false;
  for (int r = 0; r < fan0; r++) {
    int bi = -1;
    float bv = -1.0f;
    for (int j = 0; j < 16; j++) {
      if (!picked0[j] && p0[j] > bv) { bv = p0[j]; bi = j; }
    }
    if (bi < 0) break;
    picked0[bi] = true;
    float q = clamp(bv, 1e-6f, 1.0f - 1e-6f);
    float z = slope * metal::log(q / (1.0f - q)) + intercepts[0];
    float cp = (z >= 0.0f) ? (1.0f / (1.0f + metal::exp(-z))) : (metal::exp(z) / (1.0f + metal::exp(z)));
    c_jp[n_cand] = cp;
    c_dep[n_cand] = 1;
    c_par[n_cand] = 0;
    c_slot[n_cand] = bi;
    c_tok[n_cand] = uint(top_idx[bi]);
    c_used[n_cand] = false;
    n_cand++;
  }

  // Selected nodes: node 0 is anchor
  float sel_jp[8];
  int   sel_dep[8];
  int   sel_par[8];
  int   sel_ord[8];
  uint  sel_tok[8];
  sel_jp[0] = 1.0f; sel_dep[0] = 0; sel_par[0] = -1; sel_ord[0] = 0; sel_tok[0] = anchor[0];
  int n_sel = 1;

  while (n_sel < WW) {
    int bi = -1;
    float bv = -1.0f;
    for (int i = 0; i < n_cand; i++) {
      if (!c_used[i] && c_jp[i] > bv) { bv = c_jp[i]; bi = i; }
    }
    if (bi < 0) break;
    c_used[bi] = true;

    int par_o = c_par[bi];
    uint tok_i = c_tok[bi];
    bool dup = false;
    for (int s = 1; s < n_sel; s++) {
      if (sel_par[s] == par_o && sel_tok[s] == tok_i) { dup = true; break; }
    }
    if (dup) continue;

    int cur_oid = n_sel;
    int dep = c_dep[bi];
    int slot = c_slot[bi];
    sel_jp[cur_oid] = bv;
    sel_dep[cur_oid] = dep;
    sel_par[cur_oid] = par_o;
    sel_ord[cur_oid] = cur_oid;
    sel_tok[cur_oid] = tok_i;
    n_sel++;

    if (dep < WW - 1 && n_sel < WW) {
      const device float* row = sc16 + ((dep - 1) * 16 + slot) * 16;
      float pr[16];
      float mxr = row[0];
      for (int j = 1; j < 16; j++) mxr = max(mxr, row[j]);
      float smr = 0.0f;
      for (int j = 0; j < 16; j++) { pr[j] = metal::exp(row[j] - mxr); smr += pr[j]; }
      float invr = 1.0f / max(smr, 1e-12f);
      for (int j = 0; j < 16; j++) pr[j] *= invr;

      bool pickedr[16];
      for (int j = 0; j < 16; j++) pickedr[j] = false;
      int d_idx = min(6, dep);
      for (int r = 0; r < 3 && n_cand < MAX_C; r++) {
        int bj = -1;
        float bvr = -1.0f;
        for (int j = 0; j < 16; j++) {
          if (!pickedr[j] && pr[j] > bvr) { bvr = pr[j]; bj = j; }
        }
        if (bj < 0) break;
        pickedr[bj] = true;
        float q = clamp(bvr, 1e-6f, 1.0f - 1e-6f);
        float z = slope * metal::log(q / (1.0f - q)) + intercepts[d_idx];
        float cp = (z >= 0.0f) ? (1.0f / (1.0f + metal::exp(-z))) : (metal::exp(z) / (1.0f + metal::exp(z)));
        c_jp[n_cand] = bv * cp;
        c_dep[n_cand] = dep + 1;
        c_par[n_cand] = cur_oid;
        c_slot[n_cand] = bj;
        c_tok[n_cand] = uint(top_idx[dep * 16 + bj]);
        c_used[n_cand] = false;
        n_cand++;
      }
    }
  }

  // Choose optimal verify width w_act in [2, n_sel] maximizing E[tokens] / cost_arr[w]
  int best_w = 2;
  float exp_t = 1.0f + clamp(sel_jp[1], 0.0f, 1.0f);
  float best_rate = exp_t / max(1e-3f, cost_arr[2]);
  for (int w = 3; w <= n_sel; w++) {
    exp_t += clamp(sel_jp[w - 1], 0.0f, 1.0f);
    float rate = exp_t / max(1e-3f, cost_arr[w]);
    if (rate > best_rate) { best_rate = rate; best_w = w; }
  }
  w_act_out[0] = best_w;

  // Topological sort sel[1..best_w-1] by (depth, order)
  int topo[8];
  topo[0] = 0;
  for (int i = 1; i < best_w; i++) {
    int di = sel_dep[i];
    int oi = sel_ord[i];
    int j = i;
    while (j > 1) {
      int prev = topo[j - 1];
      if (di > sel_dep[prev] || (di == sel_dep[prev] && oi >= sel_ord[prev])) break;
      topo[j] = prev;
      j--;
    }
    topo[j] = i;
  }

  int old_to_new[8];
  for (int i = 0; i < 8; i++) old_to_new[i] = 0;
  int local_par[8];
  int local_dep[8];
  uint anc_tok = anchor[0];
  block_toks[0] = anc_tok;
  parents_out[0] = -1;
  depths_out[0] = 0;
  local_par[0] = -1;
  local_dep[0] = 0;
  for (int s = 1; s < best_w; s++) {
    int old_id = topo[s];
    old_to_new[old_id] = s;
    int ps = old_to_new[sel_par[old_id]];
    int db = sel_dep[old_id];
    block_toks[s] = sel_tok[old_id];
    parents_out[s] = ps;
    depths_out[s] = db;
    local_par[s] = ps;
    local_dep[s] = db;
  }
  for (int s = best_w; s < WW; s++) {
    block_toks[s] = anc_tok;
    parents_out[s] = 0;
    depths_out[s] = 1;
    local_par[s] = 0;
    local_dep[s] = 1;
  }

  for (int b = 0; b < WW; b++) {
    int pb = local_par[b];
    int db = local_dep[b];
    int path[8];
    int cur = b;
    for (int k = db; k >= 0; k--) {
      path[k] = cur;
      cur = (k > 0) ? local_par[cur] : -1;
    }
    conv_anc[b * 3 + 0] = (db >= 3) ? path[db - 3] : -(db + 1);
    conv_anc[b * 3 + 1] = (db >= 2) ? path[db - 2] : -(db + 2);
    conv_anc[b * 3 + 2] = (db >= 1) ? pb : -3;
    tree_meta[b * 9 + 0] = db;
    for (int k = 0; k < 8; k++) {
      tree_meta[b * 9 + 1 + k] = (k <= db) ? path[k] : 0;
    }
  }
"""


def dflash2_pack_tree_dink_gpu(
    top_idx: mx.array,
    sc0: mx.array,
    sc16: mx.array,
    anchor: mx.array,
    W_max: int,
    cost_arr: mx.array,
    prior_cfg: mx.array,
):
    """Zero-sync DFlash2 calibrated hybrid tree + MoEStepCosts dynamic width on GPU."""
    global _dflash2_pack_tree_dink_k
    WW = int(W_max)
    assert 2 <= WW <= 8
    if _dflash2_pack_tree_dink_k is None:
        _dflash2_pack_tree_dink_k = mx.fast.metal_kernel(
            name="dflash2_pack_tree_dink_gpu",
            input_names=["top_idx", "sc0", "sc16", "anchor", "cost_arr", "prior_cfg"],
            output_names=["block_toks", "parents_out", "depths_out", "conv_anc", "tree_meta", "w_act_out"],
            source=_DFLASH2_PACK_TREE_DINK_SRC,
            ensure_row_contiguous=True,
        )
    anc_u = anchor.reshape(-1)
    if anc_u.dtype != mx.uint32:
        anc_u = anc_u.astype(mx.uint32)
    return _dflash2_pack_tree_dink_k(
        inputs=[
            top_idx.reshape(-1),
            sc0.reshape(-1),
            sc16.reshape(-1),
            anc_u,
            cost_arr.reshape(-1),
            prior_cfg.reshape(-1),
        ],
        template=[("WW", WW)],
        grid=(32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(WW,), (WW,), (WW,), (WW, 3), (WW, 9), (1,)],
        output_dtypes=[mx.uint32, mx.int32, mx.int32, mx.int32, mx.int32, mx.int32],
    )
