# LOOPLING_MTP: the official "DSpark/dflare" multi-token-prediction drafter of Loopling, ported to MLX.
# See docs/MTP.md for the math and the file:line citations into the vLLM reference.
from collections import deque
import json
import os
import time

import mlx.core as mx
import numpy as np
import mlx.nn as nn

from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.inkling2.rows.qmv4 import (
    draft_markov_pack_gpu,
    markov_bias_argmax_4b,
    markov_bias_topk4_4b,
    markov_bias_topk4_beam_4b,
    markov_shortlist_topk4_beam_f16,
    markov_shortlist_topk4_f16,
    pack_tree_gpu,
    pack_tree_gpu_multi,
    pack_tree_gpu_multi_dink,
    qkv_norm_rope_f16,
    qmv4_fp16_rows,
    qmv4_fp16_swiglu_rows,
    qmv4_fp16_topk2_rows,
    qmv4_rows,
    sdpa_draft_gpu,
    sdpa_draft_qkv_gpu,
    topk2_chunks_f16,
    tree_walk_commit_meta_gpu,
)

MTP_DIR = os.environ.get("LOOPLING_MTP_DIR", os.path.expanduser("~/loopling/models/official-mtp"))
EPS = 1e-6

_pos_arr_cache: dict = {}


def _pos_arr(pos: int) -> mx.array:
    a = _pos_arr_cache.get(pos)
    if a is None:
        if len(_pos_arr_cache) >= 512:
            _pos_arr_cache.clear()
        a = _pos_arr_cache[pos] = mx.array([pos], mx.int32)
    return a


_swiglu_bf16 = mx.compile(lambda g, u: nn.silu(g) * u)


@mx.compile
def _rnn_step_c(ctx_i, state, w1_prev, w_recT):
    inp = mx.concatenate([state, w1_prev], axis=-1)
    raw = (ctx_i + inp.astype(mx.float32) @ w_recT).astype(mx.bfloat16)
    r = state.shape[-1]
    gate = mx.sigmoid(raw[:, :r])
    cand = mx.tanh(raw[:, r : 2 * r])
    new_state = gate * state + (1 - gate) * cand
    latent = mx.tanh(raw[:, 2 * r :])
    return new_state, latent


@mx.compile
def _mix_c(feats, a_row, norm_w):
    m = (feats.astype(mx.float32) * a_row).sum(axis=2)
    return mx.fast.rms_norm(m, norm_w.astype(mx.float32), EPS).astype(mx.bfloat16)


@mx.compile
def _mix_f16_c(feats, a_row, norm_w):
    m = (feats.astype(mx.float32) * a_row).sum(axis=2)
    return mx.fast.rms_norm(m, norm_w.astype(mx.float32), EPS).astype(mx.float16)


@mx.compile
def _add_rms_c(res, delta, ln_w):
    r = res + delta
    return r, mx.fast.rms_norm(r, ln_w, EPS)


@mx.compile
def _emb_rms_c(emb, mask_embed, ln0_w):
    x = mx.concatenate([emb, mx.broadcast_to(mask_embed.reshape(1, 1, -1), (1, 7, emb.shape[-1]))], axis=1)
    return x, mx.fast.rms_norm(x, ln0_w, EPS)


_emb_rms_f16_cache: dict = {}


def _get_emb_rms_f16(n_slots: int):
    fn = _emb_rms_f16_cache.get(n_slots)
    if fn is None:
        n_mask = n_slots - 1

        @mx.compile
        def _fn(emb, mask_embed, ln0_w):
            x = mx.concatenate(
                [emb.astype(mx.float16), mx.broadcast_to(mask_embed.reshape(1, 1, -1), (1, n_mask, emb.shape[-1]))],
                axis=1,
            )
            return x, mx.fast.rms_norm(x, ln0_w, EPS)

        fn = _emb_rms_f16_cache[n_slots] = _fn
    return fn


@mx.compile
def _ctx_all_c(hid, f_feat, a_mk, mk_dn, w_h_f32, w_f_f32, rnn_bias):
    m = (f_feat.astype(mx.float32) * a_mk).sum(axis=2)
    f_rnn = mx.fast.rms_norm(m, mk_dn.astype(mx.float32), EPS).astype(mx.bfloat16)[0]
    return hid.astype(mx.float32) @ w_h_f32 + f_rnn.astype(mx.float32) @ w_f_f32 + rnn_bias


@mx.compile
def _ctx_all_f16_c(hid, f_feat, a_mk, mk_dn, w_h_f16, w_f_f16, rnn_bias_f16):
    m = (f_feat.astype(mx.float32) * a_mk).sum(axis=2)
    f_rnn = mx.fast.rms_norm(m, mk_dn.astype(mx.float32), EPS).astype(mx.float16)[0]
    return (hid @ w_h_f16 + f_rnn @ w_f_f16 + rnn_bias_f16).astype(mx.float32)


@mx.compile
def _beam_top2_c(jp_f, prob_d, tok_d, st_d):
    cjp = (jp_f[:, None] * prob_d).reshape(-1)
    ctok = tok_d.reshape(-1)
    top = mx.argsort(-cjp)[:2]
    return cjp, ctok, ctok[top], cjp[top], st_d[top // 4], top


def rms(x, w=None):
    return mx.fast.rms_norm(x, w, EPS)


class Drafter:
    """3-layer block drafter (block 8) + Markov (rnn_hidden) head. bf16, batch 1."""

    def __init__(self, target, mtp_dir=MTP_DIR, tap_names=None, ones_layer0_norm=True):
        cfg = json.load(open(os.path.join(mtp_dir, "config.json")))["mtp_config"]
        self.cfg = cfg
        self.H = cfg["hidden_size"]
        self.nl = cfg["num_hidden_layers"]
        self.nh, self.nkv, self.hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
        self.block = cfg["block_size"]
        self.window = cfg["sliding_window_size"]
        self.local = set(cfg["local_layer_ids"])
        self.rank = cfg["markov_rank"]
        self.theta = cfg["rope_theta"]
        self.target = target
        w = mx.load(os.path.join(mtp_dir, "mtp.safetensors"))
        self.w = w
        ntap = len(cfg["target_layer_ids"])
        self.alpha = mx.softmax(w["fuse_weight"].astype(mx.float32), axis=-1)  # (nl+1, ntap)
        self.mask_embed = w["mask_embed"]
        self.final_norm = w["norm.weight"]
        self.dnorm = []
        for i in range(self.nl):
            k = "layers.%d.dflare_norm.weight" % i
            # layers.0.dflare_norm is not in the checkpoint; the reference's _RMSNorm defaults to ones.
            self.dnorm.append(w[k] if k in w else mx.ones((self.H,), mx.bfloat16))
        _a = os.environ.get("MTP_L0NORM")  # ablation only
        if _a == "l1":
            self.dnorm[0] = self.dnorm[1]
        elif _a == "l2":
            self.dnorm[0] = self.dnorm[2]
        self.mk_dnorm = w["markov_head.dflare_norm.weight"]
        r = self.rank
        rp = w["markov_head.rnn_proj.weight"]  # (3r, 2r+2H)
        self.w_rec = rp[:, : 2 * r]
        self.w_ctx = rp[:, 2 * r :]
        self.rnn_bias = w["markov_head.rnn_proj.bias"].astype(mx.float32)
        self.w1 = w["markov_head.markov_w1.weight"]
        # fold hidden_out into the target lm_head: bias_logits = lm_head(hidden_out(latent))
        lm = target.lm_head
        ho_w, ho_b = w["markov_head.hidden_out.weight"], w["markov_head.hidden_out.bias"]  # (H,r), (H,)
        self.fold_w = lm(ho_w.T[None]).astype(mx.bfloat16)[0].T  # (V, r)  (columns of hidden_out through lm_head)
        self.fold_b = lm(ho_b[None, None]).astype(mx.bfloat16)[0, 0]  # (V,)
        mx.eval(self.fold_w, self.fold_b)
        self.fold_wT = mx.contiguous(self.fold_w.T)  # (r, V)
        mx.eval(self.fold_wT)
        self.softcap = target.args.text.final_logit_softcapping
        self.vocab = target.args.text.unpadded_vocab_size
        self.layer_w = [
            {k.split(".", 2)[2]: v for k, v in w.items() if k.startswith("layers.%d." % i)} for i in range(self.nl)
        ]
        self.reset()

    # ---- context K/V (one entry per committed target position) ----
    def reset(self):
        self.n = 0
        self.ck = [None] * self.nl
        self.cv = [None] * self.nl

    def truncate(self, n):
        self.n = n

    def mix(self, feats, row, norm_w):
        """feats (B,T,ntap,H) -> RMSNorm(sum_t softmax(fuse_weight)[row,t] * feats[..., t, :]) (fp32 math) -> bf16."""
        a = self.alpha[row].reshape(1, 1, -1, 1)
        m = (feats.astype(mx.float32) * a).sum(axis=2)
        return rms(m, norm_w.astype(mx.float32)).astype(mx.bfloat16)

    def append_ctx(self, feats):
        """feats (1,T,ntap,H): target features of the next T committed positions (positions self.n ...)."""
        T = feats.shape[1]
        for l in range(self.nl):
            lw = self.layer_w[l]
            c = self.mix(feats, l, self.dnorm[l])
            k = (c @ lw["self_attn.k_proj_ctx.weight"].T).reshape(1, T, self.nkv, self.hd)
            v = (c @ lw["self_attn.v_proj_ctx.weight"].T).reshape(1, T, self.nkv, self.hd)
            k = rms(k, lw["self_attn.k_norm_ctx.weight"]).transpose(0, 2, 1, 3)
            v = v.transpose(0, 2, 1, 3)
            k = mx.fast.rope(k, self.hd, traditional=False, base=self.theta, scale=1.0, offset=self.n)
            if self.ck[l] is None or self.n + T > self.ck[l].shape[2]:
                cap = ((self.n + T + 1023) // 1024 + 1) * 1024
                nk = mx.zeros((1, self.nkv, cap, self.hd), mx.bfloat16)
                nv = mx.zeros((1, self.nkv, cap, self.hd), mx.bfloat16)
                if self.ck[l] is not None:
                    nk[:, :, : self.n] = self.ck[l][:, :, : self.n]
                    nv[:, :, : self.n] = self.cv[l][:, :, : self.n]
                self.ck[l], self.cv[l] = nk, nv
            self.ck[l][:, :, self.n : self.n + T] = k
            self.cv[l][:, :, self.n : self.n + T] = v
        self.n += T

    # ---- one draft block ----
    def draft(self, anchor, f_feat, return_logits=False):
        """anchor: int array (1,) token at position P=self.n; f_feat (1,1,ntap,H): features of position P-1.
        Returns (7,) draft tokens for positions P+1..P+7 (lazy)."""
        B, P = self.block, self.n
        emb = self.target.model.language_model.embed(anchor.reshape(1, 1))  # (1,1,H)
        x = mx.concatenate([emb, mx.broadcast_to(self.mask_embed.reshape(1, 1, -1), (1, B - 1, self.H))], axis=1)
        res = x
        h = rms(x, self.layer_w[0]["input_layernorm.weight"])
        for l in range(self.nl):
            lw = self.layer_w[l]
            if l > 0:
                h = rms(res, lw["input_layernorm.weight"])
            q = (h @ lw["self_attn.q_proj.weight"].T).reshape(1, B, self.nh, self.hd)
            k = (h @ lw["self_attn.k_proj.weight"].T).reshape(1, B, self.nkv, self.hd)
            v = (h @ lw["self_attn.v_proj.weight"].T).reshape(1, B, self.nkv, self.hd)
            q = rms(q, lw["self_attn.q_norm.weight"]).transpose(0, 2, 1, 3)
            k = rms(k, lw["self_attn.k_norm.weight"]).transpose(0, 2, 1, 3)
            v = v.transpose(0, 2, 1, 3)
            q = mx.fast.rope(q, self.hd, traditional=False, base=self.theta, scale=1.0, offset=P)
            k = mx.fast.rope(k, self.hd, traditional=False, base=self.theta, scale=1.0, offset=P)
            s = max(0, P - self.window) if l in self.local else 0
            K = mx.concatenate([self.ck[l][:, :, s:P], k], axis=2) if P > s else k
            V = mx.concatenate([self.cv[l][:, :, s:P], v], axis=2) if P > s else v
            o = mx.fast.scaled_dot_product_attention(q, K, V, scale=self.hd**-0.5, mask=None)
            o = o.transpose(0, 2, 1, 3).reshape(1, B, -1) @ lw["self_attn.o_proj.weight"].T
            res = res + o
            h2 = rms(res, lw["post_attention_layernorm.weight"])
            g = h2 @ lw["mlp.gate_proj.weight"].T
            u = h2 @ lw["mlp.up_proj.weight"].T
            res = res + (nn.silu(g) * u) @ lw["mlp.down_proj.weight"].T
        hid = rms(res, self.final_norm)[0, 1:]  # (7,H) slots 1..7
        base = self.target.lm_head(hid)  # (7,V) raw lm_head logits (pre-softcap)
        # f_rnn: mix of last committed position's features with the Markov dflare_norm
        f_rnn = self.mix(f_feat, self.nl, self.mk_dnorm)[0]  # (1,H)
        ctx = (
            hid.astype(mx.float32) @ self.w_ctx[:, : self.H].T.astype(mx.float32)
            + f_rnn.astype(mx.float32) @ self.w_ctx[:, self.H :].T.astype(mx.float32)
            + self.rnn_bias
        )  # (7,3r)
        r = self.rank
        state = mx.zeros((1, r), mx.bfloat16)
        prev = anchor.reshape(1)
        toks = []
        for i in range(B - 1):
            inp = mx.concatenate([state, self.w1[prev]], axis=-1)
            raw = (ctx[i : i + 1] + inp.astype(mx.float32) @ self.w_rec.T.astype(mx.float32)).astype(mx.bfloat16)
            gate = mx.sigmoid(raw[:, :r])
            cand = mx.tanh(raw[:, r : 2 * r])
            state = gate * state + (1 - gate) * cand
            latent = mx.tanh(raw[:, 2 * r :])
            bias = latent @ self.fold_wT + self.fold_b
            lg = base[i : i + 1] + (0 if os.environ.get("MTP_NOBIAS") else bias)
            lg = self.softcap * mx.tanh(lg.astype(mx.float32) / self.softcap)
            prev = mx.argmax(lg[:, : self.vocab], axis=-1)
            toks.append(prev)
        return mx.concatenate(toks)


class FastDrafter:
    """Quantized 4-bit MTP drafter + 4-bit SLC-resident Markov head (drafter need not be bit-exact)."""

    def __init__(self, target, mtp_dir=MTP_DIR):
        cfg = json.load(open(os.path.join(mtp_dir, "config.json")))["mtp_config"]
        self.cfg = cfg
        self.H = cfg["hidden_size"]
        self.nl = cfg["num_hidden_layers"]
        self.nh, self.nkv, self.hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
        self.block = cfg["block_size"]
        self.window = cfg["sliding_window_size"]
        self.local = set(cfg["local_layer_ids"])
        self.rank = cfg["markov_rank"]
        self.theta = cfg["rope_theta"]
        self.target = target
        w = mx.load(os.path.join(mtp_dir, "mtp.safetensors"))
        self.alpha = mx.softmax(w["fuse_weight"].astype(mx.float32), axis=-1)
        self.mask_embed = w["mask_embed"].astype(mx.float16)
        self.final_norm = w["norm.weight"].astype(mx.float16)
        self.dnorm = []
        for i in range(self.nl):
            k = "layers.%d.dflare_norm.weight" % i
            self.dnorm.append(w[k].astype(mx.float16) if k in w else mx.ones((self.H,), mx.float16))
        self.mk_dnorm = w["markov_head.dflare_norm.weight"].astype(mx.float16)
        r = self.rank
        rp = w["markov_head.rnn_proj.weight"]
        self.w_recT_f32 = mx.contiguous(rp[:, : 2 * r].T.astype(mx.float32))
        self.w_recT_f16 = mx.contiguous(self.w_recT_f32.astype(mx.float16))
        self.w_ctx_h_f16 = mx.contiguous(rp[:, 2 * r : 2 * r + self.H].T.astype(mx.float16))
        self.w_ctx_f_f16 = mx.contiguous(rp[:, 2 * r + self.H :].T.astype(mx.float16))
        self.rnn_bias_f16 = w["markov_head.rnn_proj.bias"].astype(mx.float16)
        self.w1 = w["markov_head.markov_w1.weight"]
        self.softcap = target.args.text.final_logit_softcapping
        self.vocab = target.args.text.unpadded_vocab_size

        # Fold hidden_out through target lm_head (before copying); keep fp16 fold_w/b for shortlist Markov + 4b for fallback.
        lm = target.lm_head
        ho_w, ho_b = w["markov_head.hidden_out.weight"], w["markov_head.hidden_out.bias"]
        fold_w_bf = lm(ho_w.T[None]).astype(mx.bfloat16)[0].T[: self.vocab]
        self.fold_w_f16 = mx.contiguous(fold_w_bf.astype(mx.float16))
        self.fold_b = mx.contiguous(lm(ho_b[None, None]).astype(mx.bfloat16)[0, 0][: self.vocab])
        self.fold_b_f16 = mx.contiguous(self.fold_b.astype(mx.float16))
        self.fw_w, self.fw_s, self.fw_b = mx.quantize(fold_w_bf, group_size=64, bits=4, mode="affine")

        # Separate 4-bit gs=64 fp16 copy of lm_head[:vocab] (never mutates target.lm_head)
        w_lm_full = mx.dequantize(
            lm["weight"], lm["scales"], lm["biases"], group_size=lm.group_size, bits=lm.bits, mode=lm.mode
        )[: self.vocab].astype(mx.float16)
        self.lm_w, self.lm_s, self.lm_b = mx.quantize(w_lm_full, group_size=64, bits=4, mode="affine")

        # Quantize & fuse 3-layer MTP weights to 4-bit gs=64 with fp16 scales/biases & norms
        self.qlw = []
        for l in range(self.nl):
            p = f"layers.{l}."
            qkv_w = mx.concatenate(
                [w[p + "self_attn.q_proj.weight"], w[p + "self_attn.k_proj.weight"], w[p + "self_attn.v_proj.weight"]],
                axis=0,
            ).astype(mx.float16)
            gate_w = w[p + "mlp.gate_proj.weight"].astype(mx.float16)
            up_w = w[p + "mlp.up_proj.weight"].astype(mx.float16)
            gu_w = mx.stack([gate_w, up_w], axis=1).reshape(-1, self.H)
            kvc_w = mx.concatenate([w[p + "self_attn.k_proj_ctx.weight"], w[p + "self_attn.v_proj_ctx.weight"]], axis=0).astype(mx.float16)
            self.qlw.append(
                {
                    "qkv": mx.quantize(qkv_w, group_size=64, bits=4, mode="affine"),
                    "o": mx.quantize(w[p + "self_attn.o_proj.weight"].astype(mx.float16), group_size=64, bits=4, mode="affine"),
                    "gu": mx.quantize(gu_w, group_size=64, bits=4, mode="affine"),
                    "down": mx.quantize(w[p + "mlp.down_proj.weight"].astype(mx.float16), group_size=64, bits=4, mode="affine"),
                    "kv_ctx": mx.quantize(kvc_w, group_size=64, bits=4, mode="affine"),
                    "in_ln": w[p + "input_layernorm.weight"].astype(mx.float16),
                    "post_ln": w[p + "post_attention_layernorm.weight"].astype(mx.float16),
                    "q_norm": w[p + "self_attn.q_norm.weight"].astype(mx.float16),
                    "k_norm": w[p + "self_attn.k_norm.weight"].astype(mx.float16),
                    "k_norm_ctx": w[p + "self_attn.k_norm_ctx.weight"].astype(mx.float16),
                }
            )
        mx.eval(
            self.mask_embed, self.final_norm, self.dnorm, self.mk_dnorm,
            self.fold_w_f16, self.fold_b_f16,
            self.fw_w, self.fw_s, self.fw_b, self.fold_b,
            self.lm_w, self.lm_s, self.lm_b,
            self.w_recT_f32, self.w_recT_f16, self.w_ctx_h_f16, self.w_ctx_f_f16, self.rnn_bias_f16,
            self.qlw,
        )
        nh, nkv, hd, q_dim, kv_dim = self.nh, self.nkv, self.hd, self.nh * self.hd, self.nkv * self.hd
        self._qkv_prep_cache = {}

        def _get_qkv_prep(n_slots: int):
            fn = self._qkv_prep_cache.get(n_slots)
            if fn is None:
                @mx.compile
                def _qkv_prep(qkv, qn_w, kn_w):
                    q = mx.fast.rms_norm(qkv[:, :q_dim].reshape(1, n_slots, nh, hd), qn_w, EPS).transpose(0, 2, 1, 3)
                    k = mx.fast.rms_norm(qkv[:, q_dim : q_dim + kv_dim].reshape(1, n_slots, nkv, hd), kn_w, EPS).transpose(0, 2, 1, 3)
                    v = qkv[:, q_dim + kv_dim :].reshape(1, n_slots, nkv, hd).transpose(0, 2, 1, 3)
                    return q, k, v

                fn = self._qkv_prep_cache[n_slots] = _qkv_prep
            return fn

        self._get_qkv_prep = _get_qkv_prep
        self.reset()

    def reset(self):
        self.n = 0
        self.ck = [None] * self.nl
        self.cv = [None] * self.nl

    def truncate(self, n):
        self.n = n

    def _ensure_cap(self, need):
        if self.ck[0] is not None and need <= self.ck[0].shape[2]:
            return
        cap = max(2048, ((need + 1023) // 1024 + 1) * 1024)
        for l in range(self.nl):
            nk = mx.zeros((1, self.nkv, cap, self.hd), mx.float16)
            nv = mx.zeros((1, self.nkv, cap, self.hd), mx.float16)
            if self.ck[l] is not None and self.n > 0:
                bk, bv = self.ck[l], self.cv[l]
                self.ck[l] = self.cv[l] = None
                nk = mx.slice_update(nk, bk[:, :, : self.n, :], _pos_arr(0), axes=(2,))
                nv = mx.slice_update(nv, bv[:, :, : self.n, :], _pos_arr(0), axes=(2,))
            else:
                self.ck[l] = self.cv[l] = None
            self.ck[l], self.cv[l] = nk, nv

    def mix(self, feats, row, norm_w):
        return _mix_f16_c(feats, self.alpha[row].reshape(1, 1, -1, 1), norm_w)

    def _proj4(self, x2d, qt):
        w, s, b = qt
        if x2d.shape[0] <= 16:
            return qmv4_fp16_rows(x2d, w, s, b)
        return mx.quantized_matmul(x2d, w, scales=s, biases=b, transpose=True, group_size=64, bits=4, mode="affine")

    def append_ctx_gpu(self, feats, pos_arr):
        """feats: (1, T, ntap, H). Append next T committed positions at GPU `pos_arr (1,) i32`."""
        T = feats.shape[1]
        rope_off = pos_arr[0]
        for l in range(self.nl):
            lw = self.qlw[l]
            c = self.mix(feats, l, self.dnorm[l]).reshape(T, self.H)
            kv = self._proj4(c, lw["kv_ctx"])
            k = kv[:, : self.nkv * self.hd].reshape(1, T, self.nkv, self.hd)
            v = kv[:, self.nkv * self.hd :].reshape(1, T, self.nkv, self.hd)
            k = rms(k, lw["k_norm_ctx"]).transpose(0, 2, 1, 3)
            v = v.transpose(0, 2, 1, 3)
            k = mx.fast.rope(k, self.hd, traditional=False, base=self.theta, scale=1.0, offset=rope_off)
            bk, bv = self.ck[l], self.cv[l]
            self.ck[l] = self.cv[l] = None
            self.ck[l] = mx.slice_update(bk, k, pos_arr, axes=(2,))
            self.cv[l] = mx.slice_update(bv, v, pos_arr, axes=(2,))

    def append_ctx(self, feats):
        """feats: (1, T, ntap, H). Append next T committed positions to drafter KV cache."""
        T = feats.shape[1]
        self._ensure_cap(self.n + T + self.block)
        self.append_ctx_gpu(feats, _pos_arr(self.n))
        self.n += T

    def _body(self, anchor, n_slots=None, pos_arr=None):
        """Run 3-layer fp16 MTP body for `B = n_slots` slots; returns final hidden `(B, H)` fp16."""
        B = self.block if n_slots is None else int(n_slots)
        if pos_arr is None:
            P = self.n
            self._ensure_cap(P + B)
            pos_arr = _pos_arr(P)
        use_fuse_qkv = os.environ.get("LOOPLING_MTP_FUSE_QKV", "1") not in ("0", "")
        use_fuse_sdpa_qkv = os.environ.get("LOOPLING_MTP_FUSE_SDPA_QKV", "0") not in ("0", "")
        emb = self.target.model.language_model.embed(anchor.reshape(1, 1))
        res, h = _get_emb_rms_f16(B)(emb, self.mask_embed, self.qlw[0]["in_ln"])
        qkv_prep = None if (use_fuse_qkv or use_fuse_sdpa_qkv) else self._get_qkv_prep(B)
        rope_off = None if (use_fuse_qkv or use_fuse_sdpa_qkv) else pos_arr[0]
        for l in range(self.nl):
            lw = self.qlw[l]
            qkv = mx.quantized_matmul(
                h.reshape(B, self.H), lw["qkv"][0], scales=lw["qkv"][1], biases=lw["qkv"][2],
                transpose=True, group_size=64, bits=4, mode="affine",
            )
            win = self.window if l in self.local else 0
            if use_fuse_sdpa_qkv:
                o = sdpa_draft_qkv_gpu(
                    qkv, lw["q_norm"], lw["k_norm"], self.ck[l], self.cv[l], pos_arr,
                    nh=self.nh, nkv=self.nkv, hd=self.hd, theta=self.theta, scale=self.hd**-0.5, window=win,
                )
            else:
                if use_fuse_qkv:
                    q, k, v = qkv_norm_rope_f16(
                        qkv, lw["q_norm"], lw["k_norm"], pos_arr, self.nh, self.nkv, self.hd, self.theta
                    )
                else:
                    q, k, v = qkv_prep(qkv, lw["q_norm"], lw["k_norm"])
                    q = mx.fast.rope(q, self.hd, traditional=False, base=self.theta, scale=1.0, offset=rope_off)
                    k = mx.fast.rope(k, self.hd, traditional=False, base=self.theta, scale=1.0, offset=rope_off)
                o = sdpa_draft_gpu(q, self.ck[l], self.cv[l], k, v, pos_arr, scale=self.hd**-0.5, window=win)
            o = mx.quantized_matmul(
                o, lw["o"][0], scales=lw["o"][1], biases=lw["o"][2],
                transpose=True, group_size=64, bits=4, mode="affine",
            ).reshape(1, B, self.H)
            res, h2 = _add_rms_c(res, o, lw["post_ln"])
            act = qmv4_fp16_swiglu_rows(h2.reshape(B, self.H), *lw["gu"])
            down = mx.quantized_matmul(
                act, lw["down"][0], scales=lw["down"][1], biases=lw["down"][2],
                transpose=True, group_size=64, bits=4, mode="affine",
            ).reshape(1, B, self.H)
            if l + 1 == self.nl:
                _, h_final = _add_rms_c(res, down, self.final_norm)
                return h_final[0]
            res, h = _add_rms_c(res, down, self.qlw[l + 1]["in_ln"])

    def _lm_base(self, hid):
        if int(hid.shape[0]) <= 3 and os.environ.get("LOOPLING_MTP_LM_QMM", "1") not in ("0", ""):
            return mx.quantized_matmul(
                hid, self.lm_w, scales=self.lm_s, biases=self.lm_b,
                transpose=True, group_size=64, bits=4, mode="affine",
            )
        return qmv4_fp16_rows(hid, self.lm_w, self.lm_s, self.lm_b)

    def draft(self, anchor, f_feat, n_draft=7):
        """Draft `n_draft` (1..7) chain tokens from `anchor` (1,) at position P=self.n and `f_feat` (1,1,ntap,H)."""
        hid = self._body(anchor, n_slots=1 + n_draft)[1:]
        base = self._lm_base(hid)
        ctx = _ctx_all_f16_c(
            hid, f_feat, self.alpha[self.nl].reshape(1, 1, -1, 1),
            self.mk_dnorm, self.w_ctx_h_f16, self.w_ctx_f_f16, self.rnn_bias_f16,
        )
        state = mx.zeros((1, self.rank), mx.bfloat16)
        prev = anchor.reshape(1)
        toks = []
        for i in range(n_draft):
            state, latent = _rnn_step_c(ctx[i : i + 1], state, self.w1[prev], self.w_recT_f32)
            prev = markov_bias_argmax_4b(latent, self.fw_w, self.fw_s, self.fw_b, self.fold_b, base[i], self.vocab)
            toks.append(prev)
        return mx.concatenate(toks)

    def _draft_candidates(self, anchor, f_feat, max_depth=5, beam_w=2, pos_arr=None):
        assert 1 <= max_depth <= self.block - 1 and beam_w == 2
        hid = self._body(anchor, n_slots=1 + max_depth, pos_arr=pos_arr)[1:]
        base = self._lm_base(hid)
        ctx = _ctx_all_f16_c(
            hid, f_feat, self.alpha[self.nl].reshape(1, 1, -1, 1),
            self.mk_dnorm, self.w_ctx_h_f16, self.w_ctx_f_f16, self.rnn_bias_f16,
        )
        r = self.rank
        st0, lat0 = _rnn_step_c(ctx[0:1], mx.zeros((1, r), mx.bfloat16), self.w1[anchor.reshape(1)], self.w_recT_f32)
        use_sl = os.environ.get("LOOPLING_MTP_SHORTLIST", "1") not in ("0", "")
        if use_sl:
            sl_sp = int(os.environ.get("LOOPLING_MTP_SL_SP", "256"))
            temp = float(os.environ.get("LOOPLING_MTP_TEMP", "1.7"))
            top_val, top_idx = topk2_chunks_f16(base, self.fold_b_f16, sp=sl_sp)
            tok0, prob0 = markov_shortlist_topk4_f16(lat0, top_val[0], top_idx[0], self.fold_w_f16, self.softcap, temp)
        else:
            tok0, prob0 = markov_bias_topk4_4b(lat0, self.fw_w, self.fw_s, self.fw_b, self.fold_b, base[0], self.vocab, self.softcap)
        triples = []
        if max_depth == 1:
            return tok0[0], prob0[0], triples

        tok_f = tok0[0, :beam_w]
        jp_f = prob0[0, :beam_w]
        st_f = mx.broadcast_to(st0, (beam_w, r))
        pidx_f = mx.array([1, 2], mx.int32)
        for d in range(1, max_depth):
            ctx_d = mx.broadcast_to(ctx[d : d + 1], (beam_w, 3 * r))
            st_d, lat_d = _rnn_step_c(ctx_d, st_f, self.w1[tok_f], self.w_recT_f32)
            pool_base = 1 + 4 + (d - 1) * 8
            if use_sl:
                ctok, cjp, cpar, tok_f, jp_f, st_f, pidx_f = markov_shortlist_topk4_beam_f16(
                    lat_d, top_val[d], top_idx[d], self.fold_w_f16,
                    jp_f, pidx_f, st_d, self.softcap, pool_base, temp,
                )
            else:
                ctok, cjp, cpar, tok_f, jp_f, st_f, pidx_f = markov_bias_topk4_beam_4b(
                    lat_d, self.fw_w, self.fw_s, self.fw_b, self.fold_b, base[d],
                    jp_f, pidx_f, st_d, self.vocab, self.softcap, pool_base,
                )
            triples.append((ctok, cjp, cpar))
        return tok0[0], prob0[0], triples

    def draft_tree(self, anchor, f_feat, max_depth=5, beam_w=2, pos_arr=None):
        """Zero-sync GPU beam-tree draft up to `max_depth` depths; returns flat `(all_toks, all_jprobs, all_pars)`."""
        t0, p0, triples = self._draft_candidates(anchor, f_feat, max_depth=max_depth, beam_w=beam_w, pos_arr=pos_arr)
        if not triples:
            return t0, p0, mx.zeros((4,), mx.int32)
        all_toks = [t0] + [c[0] for c in triples]
        all_jprobs = [p0] + [c[1] for c in triples]
        all_pars = [mx.zeros((4,), mx.int32)] + [c[2] for c in triples]
        return mx.concatenate(all_toks), mx.concatenate(all_jprobs), mx.concatenate(all_pars)

    def draft_and_pack_tree(self, anchor, f_feat, max_depth, W, beam_w=2, pos_arr=None):
        """Draft + pack dynamic W-node tree in one pass using fused kernels."""
        if (
            beam_w == 2
            and os.environ.get("LOOPLING_MTP_SHORTLIST", "1") not in ("0", "")
            and os.environ.get("LOOPLING_MTP_FUSE_TAIL", "0") not in ("0", "")
        ):
            assert 1 <= max_depth <= self.block - 1
            if pos_arr is None:
                P = self.n
                self._ensure_cap(P + 1 + max_depth)
                pos_arr = _pos_arr(P)
            hid = self._body(anchor, n_slots=1 + max_depth, pos_arr=pos_arr)[1:]
            sl_sp = int(os.environ.get("LOOPLING_MTP_SL_SP", "256"))
            temp = float(os.environ.get("LOOPLING_MTP_TEMP", "1.7"))
            if os.environ.get("LOOPLING_MTP_FUSE_LM_TOPK", "0") not in ("0", "") and int(hid.shape[0]) <= 5:
                top_val, top_idx = qmv4_fp16_topk2_rows(hid, self.lm_w, self.lm_s, self.lm_b, self.fold_b_f16, sp=sl_sp)
            else:
                base = self._lm_base(hid)
                top_val, top_idx = topk2_chunks_f16(base, self.fold_b_f16, sp=sl_sp)
            ctx = _ctx_all_f16_c(
                hid, f_feat, self.alpha[self.nl].reshape(1, 1, -1, 1),
                self.mk_dnorm, self.w_ctx_h_f16, self.w_ctx_f_f16, self.rnn_bias_f16,
            )
            return draft_markov_pack_gpu(
                ctx, self.w1, self.w_recT_f16, top_val, top_idx, self.fold_w_f16,
                anchor, max_depth, W, self.softcap, temp,
            )
        t0, p0, triples = self._draft_candidates(anchor, f_feat, max_depth=max_depth, beam_w=beam_w, pos_arr=pos_arr)
        return pack_tree_gpu_multi(t0, p0, triples, anchor, W)

    def draft_and_pack_tree_dink(
        self, anchor, f_feat, cost_arr, dink_cfg, *, max_depth=3, W_max=4, beam_w=2, pos_arr=None
    ):
        """Zero-sync Beam-2 tree draft + on-GPU Dinkelbach verify-width selection `w_act in [2, W_max]`."""
        t0, p0, triples = self._draft_candidates(
            anchor, f_feat, max_depth=max_depth, beam_w=beam_w, pos_arr=pos_arr
        )
        return pack_tree_gpu_multi_dink(t0, p0, triples, anchor, W_max, cost_arr, dink_cfg)


_FD_INST: dict = {}


def _get_fast_drafter(model, mtp_dir):
    lm_w = model.lm_head["weight"]
    key = (id(model), mtp_dir)
    entry = _FD_INST.get(key)
    if entry is not None and entry[0] is lm_w:
        dr = entry[1]
        dr.reset()
        return dr
    dr = FastDrafter(model, mtp_dir=mtp_dir)
    _FD_INST[key] = (lm_w, dr)
    return dr


# ---- cache rollback for the verify pass ----
def prepare_record(cache):
    for c in cache:
        c[1].record = True


def rollback(cache, L, keep):
    """Cache holds L extra tokens (a verify pass); keep only the first `keep` of them."""
    n = L - keep
    if n == 0:
        return
    for c in cache:
        kv, conv = c[0], c[1]
        if isinstance(kv, KVCache):
            kv.offset -= n
        else:  # RotatingKVCache after a multi-token (concat) update: buffer is in temporal order
            sz = kv.keys.shape[2] - n
            kv.keys, kv.values = kv.keys[..., :sz, :], kv.values[..., :sz, :]
            kv.offset -= n
            kv._idx = sz
        for i, win in enumerate(conv.windows):
            K1 = win.shape[1] - L
            conv[i] = win[:, keep : keep + K1]


def parse_taps(spec, passes):
    return [t.strip() for t in spec.split(",")]


def default_taps(passes):
    if passes == 1:
        return "i5,i10,i17,F"
    return "i5,i23,i41,F"  # literal target_layer_ids; also best at 5 passes (docs/MTP.md)


def feats_from_buf(tm, taps):
    return mx.stack([tm.tap_buf[t if t in ("F", "f", "o") else int(t[1:])] for t in taps], axis=2)  # (1,T,ntap,H)


def _feats_from_dict(buf, keys):
    return mx.stack([buf[k] for k in keys], axis=1)[None]  # (1, T, ntap, H)


_TREE_TABLE_CACHE: dict = {}


def _pack_dynamic_tree(cand_toks, cand_jprobs, cand_pars, anchor_id, max_w, cost_fn, rate_w1, cal, force_w=None):
    """Select prefix-closed top-utility tree up to `max_w` slots, sorted topologically by (depth, orig_rank)."""
    from mlx_lm.models.inkling2.batchdecode import build_tree_tables

    ctok = cand_toks.tolist()
    cjp = cand_jprobs.tolist()
    cpar = cand_pars.tolist()
    n_cand = len(ctok)
    items = [
        (i + 1, ctok[i], cjp[i], cpar[i], 1 if i < 4 else 2 + (i - 4) // 8)
        for i in range(n_cand)
    ]
    ranked = sorted(items, key=lambda it: (-it[2], it[4], it[0]))
    seen = {0}
    ordered = []

    def _add(it):
        pidx = it[0]
        if pidx in seen:
            return
        par = it[3]
        if par not in seen:
            _add(items[par - 1])
        seen.add(pidx)
        ordered.append(it)

    for it in ranked:
        _add(it)

    avail = min(int(max_w), 1 + len(ordered))
    if force_w is not None:
        best_w = max(1, min(int(force_w), avail))
    else:
        best_w, best_g = 1, rate_w1
        acc_sum = 0.0
        for w in range(2, avail + 1):
            acc_sum += ordered[w - 2][2]
            g = (1.0 + cal * acc_sum) / cost_fn(w)
            if g > best_g:
                best_g, best_w = g, w

    if best_w <= 1:
        return 1, None, None, None, None, None, None, 0.0

    sel = ordered[: best_w - 1]
    sum_sel_jp = sum(it[2] for it in sel)
    sel_topo = sorted((it[4], idx, it[0], it[1], it[3]) for idx, it in enumerate(sel))

    p2s = {0: 0}
    block_toks = [int(anchor_id)]
    parents_l = [-1]
    depths_l = [0]
    children = [[] for _ in range(best_w)]
    for slot, (dep, _, pidx, tok, par_p) in enumerate(sel_topo, start=1):
        pslot = p2s[par_p]
        p2s[pidx] = slot
        block_toks.append(tok)
        parents_l.append(pslot)
        depths_l.append(dep)
        children[pslot].append((slot, tok))

    parents_t = tuple(parents_l)
    depths_t = tuple(depths_l)
    key = (parents_t, depths_t)
    tbl = _TREE_TABLE_CACHE.get(key)
    if tbl is None:
        if len(_TREE_TABLE_CACHE) >= 512:
            _TREE_TABLE_CACHE.clear()
        tbl = _TREE_TABLE_CACHE[key] = build_tree_tables(parents_t, depths_t)
    _, conv_anc, tree_meta = tbl
    return best_w, block_toks, parents_t, depths_t, conv_anc, tree_meta, children, sum_sel_jp


def mtp_exact_generate_step(
    prompt,
    model,
    *,
    max_tokens=256,
    sampler=None,
    prefill_step_size=2048,
    prompt_cache=None,
    mtp_dir=MTP_DIR,
    **kw,
):
    """Bit-exact causal-verify speculative decoding with quantized FastDrafter."""
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.models.inkling2.batchdecode import BatchDecoder, KV_STAGE_CAP, build_tree_tables

    tm = model.model.language_model
    passes = tm.args.loop_config["num_passes"]
    taps = parse_taps(os.environ.get("LOOPLING_MTP_TAPS", default_taps(passes)), passes)
    tap_keys = [t if t in ("F", "f", "o") else int(t[1:]) for t in taps]
    tap_set = {k for k in tap_keys if isinstance(k, int)}
    dr = _get_fast_drafter(model, mtp_dir)
    cache = prompt_cache if prompt_cache is not None else make_prompt_cache(model)

    stats = getattr(model, "mtp_stats", None)
    if stats is None:
        stats = model.mtp_stats = {"cycles": 0, "accepted": 0, "tokens": 0}
    stats.update(cycles=0, accepted=0, tokens=0)

    n_p = prompt.size
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
        raise RuntimeError("mtp_exact_generate_step requires BatchDecoder._b1_fast")

    # Last prompt token via bit-exact step_with_taps
    y, tap0 = dec.step_with_taps(prompt[pos:], tap_visits=tap_set)
    f0 = _feats_from_dict(tap0, tap_keys)
    dr.append_ctx(f0)
    mx.eval(y, dr.ck, dr.cv, *dec.kv_eval_targets(), dec.conv)
    last_feat = f0[:, -1:]
    y_tok = int(y.item())
    stats["cycles"] += 1
    stats["tokens"] += 1

    anchor = y
    B = dr.block
    wmode = os.environ.get("LOOPLING_MTP_W", "auto")
    use_tree = os.environ.get("LOOPLING_MTP_TREE", "1") not in ("0", "")
    fscale = (12 + 12 * passes) / 24.0

    if use_tree:
        draft_d_env = os.environ.get("LOOPLING_MTP_D")

        def d_for_w(w):
            if draft_d_env:
                return max(1, min(B - 1, int(draft_d_env)))
            return min(4, max(1, int(w) - 1))

        # Calibrated zero-sync compiled verify_w + staged commit/append_ctx + fused FastDrafter cycle latency (ms)
        base_tree = [6.60, 7.65, 8.75, 9.65, 10.60, 11.50, 12.40]
        cost_tree = lambda w: fscale * base_tree[int(w) - 2]
        cost_w1 = 4.87 * fscale
        rate_w1 = 1.0 / cost_w1
        force_w = int(wmode) if wmode.isdigit() else None
        w_init = int(os.environ.get("LOOPLING_MTP_W_INIT", "4"))
        hi_w = int(os.environ.get("LOOPLING_MTP_W_HI", "5"))
        up_thresh = float(os.environ.get("LOOPLING_MTP_UP", "2.75"))
        dn_thresh = float(os.environ.get("LOOPLING_MTP_DN", "2.60"))
        explore_every = int(os.environ.get("LOOPLING_MTP_EXPLORE", "160"))
        burst_init = int(os.environ.get("LOOPLING_MTP_BURST", "32"))
        alpha_tpc = float(os.environ.get("LOOPLING_MTP_ALPHA", "0.14"))
        ema_tpc = float(os.environ.get("LOOPLING_MTP_TPC_INIT", "2.35"))
        min_tree_cycles = int(os.environ.get("LOOPLING_MTP_MIN_TREE", "8"))
        drop_ratio = float(os.environ.get("LOOPLING_MTP_DROP_RATIO", "0.985"))
        cur_burst = burst_init
        w_cur = w_init
        tree_cycles = 0
        is_reprobe = False
        tap_check = tap_set | {"F"}

        use_pipe_ahead = (
            os.environ.get("LOOPLING_MTP_PIPE_AHEAD", "1") not in ("0", "")
            and ((passes == 1 and dec.nv == 24) or passes > 1)
            and os.environ.get("LOOPLING_COMPILE_VERIFY", "1") not in ("0", "")
        )
        use_stage = use_pipe_ahead and os.environ.get("LOOPLING_KV_STAGE", "0") not in ("0", "")
        off_cpu = int(dec.caches[0][0][0].offset)

        def _ensure_lazy_gpu_cap(rem_toks, off_cur):
            req = rem_toks + 16
            if off_cur < 248 and os.environ.get("LOOPLING_LAZY_CAP", "0") not in ("0", ""):
                req = min(req, max(16, 256 - off_cur))
            ok = dec.ensure_gpu_cap(req)
            if ok and use_stage:
                dec.init_kv_stage()
            return ok

        if use_pipe_ahead and max_tokens > 1:
            if not _ensure_lazy_gpu_cap(max_tokens + 16, off_cpu):
                use_pipe_ahead = False
                use_stage = False
            else:
                dr._ensure_cap(dr.n + max_tokens + 32)
        tau_cap = off_cpu + max_tokens + 32

        use_dyn = (
            use_pipe_ahead
            and not use_stage
            and wmode == "auto"
            and os.environ.get("LOOPLING_MTP_DYN", "1") not in ("0", "")
            and B >= 4
        )
        if use_dyn:
            W_MAX = int(os.environ.get("LOOPLING_MTP_DYN_W", "4"))
            assert 2 <= W_MAX <= 6
            D_DYN = int(os.environ.get("LOOPLING_MTP_DYN_D", str(min(3, W_MAX - 1))))
            assert 1 <= D_DYN <= min(5, B - 1)
            default_pa = "2" if passes > 1 else "3"
            PA_DEPTH = max(1, min(3, int(os.environ.get("LOOPLING_MTP_PA", default_pa))))

            costs_env = os.environ.get("LOOPLING_MTP_DYN_COSTS")
            if costs_env:
                vals = [float(x) for x in costs_env.split(",")]
                if len(vals) == W_MAX - 1:
                    costs = [0.0, 4.88 * fscale] + vals
                elif len(vals) == W_MAX:
                    costs = [0.0] + vals
                else:
                    raise ValueError(f"Expected {W_MAX - 1} or {W_MAX} costs, got {len(vals)}")
            else:
                if passes == 3:
                    base_costs = [0.0, 9.60, 13.60, 15.40, 17.60, 18.70, 20.20]
                    costs = base_costs[: W_MAX + 1]
                else:
                    base_costs = [0.0, 4.88, 7.88, 8.48, 9.38, 10.30, 11.20]
                    costs = [c * fscale for c in base_costs[: W_MAX + 1]]
            cost_arr = mx.array(costs, mx.float32)

            cal_scale = float(os.environ.get("LOOPLING_MTP_DYN_CAL", "1.25"))
            force_dyn_w = int(os.environ.get("LOOPLING_MTP_DYN_FORCE_W", "0"))
            dink_cfg = mx.array([cal_scale, float(force_dyn_w)], mx.float32)

            w1_ms = float(os.environ.get("LOOPLING_MTP_W1_MS", str(4.88 * fscale)))
            rate_w1_dyn = 1.0 / w1_ms
            alpha_dyn = float(os.environ.get("LOOPLING_MTP_DYN_ALPHA", "0.08"))
            min_tree_dyn = int(os.environ.get("LOOPLING_MTP_DYN_MIN_CYC", "20"))
            drop_ratio_dyn = float(os.environ.get("LOOPLING_MTP_DYN_DROP", "0.70"))
            dyn_burst_init = int(os.environ.get("LOOPLING_MTP_DYN_BURST", "32"))
            dyn_burst_max = int(os.environ.get("LOOPLING_MTP_DYN_BURST_MAX", "128"))

            etok_cpu = 2.4
            ems_cpu = costs[W_MAX]
            cyc_since_burst = 0
            reprobe_d = False
            cur_burst_d = dyn_burst_init
            want_burst = False

            def _launch_gpu_cycle_dyn(packed, meta_r, meta_f, dr_pos, binfo, ring_may_wrap, launch_draft=True):
                blk_c, par_c, dep_c, ca_c, tm_c, w_act_c = packed
                walk_res, kv_news, conv_ws, tap_buf = dec.verify_w_gpu(
                    blk_c,
                    meta_r=meta_r,
                    meta_f=meta_f,
                    depths_arr=dep_c,
                    conv_anc=ca_c,
                    tree_meta=tm_c,
                    max_depth=D_DYN,
                    binfo=binfo,
                    tau_cap=tau_cap,
                    tap_visits=tap_check,
                    par_arr=par_c,
                    dr_pos=dr_pos,
                    use_stage=False,
                    w_act=w_act_c,
                )
                (
                    path_pad, emitted, keep_arr, last_slot, anc_nxt,
                    slot_r_old, end_f_old, dr_pos_old, meta_r_nxt, meta_f_nxt, dr_pos_nxt,
                ) = walk_res
                eval_a = [
                    emitted, keep_arr, path_pad, last_slot, anc_nxt,
                    slot_r_old, end_f_old, dr_pos_old, meta_r_nxt, meta_f_nxt, dr_pos_nxt,
                    *kv_news,
                ]
                for cw in conv_ws:
                    eval_a.extend(cw)
                eval_a.extend(tap_buf.values())
                mx.async_eval(*eval_a)

                dec.commit_w_gpu(
                    kv_news, conv_ws, path_pad, keep_arr, last_slot, slot_r_old, end_f_old, W_MAX,
                    ring_may_wrap=ring_may_wrap,
                )
                f_w = _feats_from_dict(tap_buf, tap_keys)
                f_pad = f_w[:, path_pad]
                last_feat_nxt = f_w[:, last_slot]
                dr.append_ctx_gpu(f_pad, dr_pos_old)

                eval_b = [*dec.kv_eval_targets(staged=False)]
                for cs in dec.conv:
                    eval_b.extend(cs)
                eval_b.extend(dr.ck)
                eval_b.extend(dr.cv)
                if launch_draft:
                    packed_nxt = dr.draft_and_pack_tree_dink(
                        anc_nxt, last_feat_nxt, cost_arr, dink_cfg,
                        max_depth=D_DYN, W_max=W_MAX, pos_arr=dr_pos_nxt,
                    )
                    eval_b.extend(packed_nxt)
                else:
                    packed_nxt = None
                mx.async_eval(*eval_b)
                return (
                    emitted, keep_arr, w_act_c, anc_nxt, last_feat_nxt,
                    meta_r_nxt, meta_f_nxt, dr_pos_nxt, packed_nxt,
                )

            def _can_launch_gpu(q_len, off_base, dr_n_base):
                nonlocal off_cpu
                if n + (q_len + 1) * W_MAX > max_tokens:
                    return False, None, False
                if q_len == 0 and not dec._kv_stacked:
                    dec.ensure_gpu_cap(max_tokens - n + 32)
                    off_base = off_cpu = int(dec.caches[0][0][0].offset)
                if not dec._kv_stacked or dr.ck[0] is None:
                    return False, None, False
                end_lo = off_base + q_len
                end_hi = off_base + q_len * W_MAX
                ok_b, binfo = dec._cw_safe_bounds_range(end_lo, end_hi, D_DYN)
                if not ok_b:
                    return False, None, False
                cap_r, cap_f, max_size_r = binfo[4], binfo[5], binfo[6]
                if q_len == 0 and (end_hi + W_MAX > cap_f or (cap_r < max_size_r and end_hi + W_MAX > cap_r)):
                    dec.ensure_gpu_cap(max_tokens - n + 32)
                    ok_b, binfo = dec._cw_safe_bounds_range(end_lo, end_hi, D_DYN)
                    if not ok_b:
                        return False, None, False
                    cap_r, cap_f, max_size_r = binfo[4], binfo[5], binfo[6]
                if (
                    end_hi + W_MAX <= cap_f
                    and (cap_r == max_size_r or end_hi + W_MAX <= cap_r)
                    and dr_n_base + (q_len + 1) * W_MAX + B <= int(dr.ck[0].shape[2])
                ):
                    return True, binfo, (end_hi + W_MAX > max_size_r)
                return False, None, False

            packed_cur = None
            inflight_q = deque()
            if max_tokens - 1 >= 2:
                packed_cur = dr.draft_and_pack_tree_dink(
                    anchor, last_feat, cost_arr, dink_cfg,
                    max_depth=D_DYN, W_max=W_MAX,
                )
                mx.async_eval(*packed_cur)

            yield y_tok, None, False
            n = 1
            pending_tps = []

            while n < max_tokens:
                if not inflight_q and packed_cur is None:
                    rem = max_tokens - n
                    n_burst = min(rem, cur_burst_d) if want_burst else rem
                    y_cur, tp = dec.step_with_taps(anchor, tap_visits=tap_set)
                    mx.async_eval(y_cur, *tp.values())
                    pending_tps.append(tp)
                    for i in range(n_burst):
                        if i + 1 < n_burst:
                            y_nxt, tp_nxt = dec.step_with_taps(y_cur, tap_visits=tap_set)
                            mx.async_eval(y_nxt, *tp_nxt.values())
                            pending_tps.append(tp_nxt)
                        else:
                            rem_after = max_tokens - (n + 1)
                            if want_burst and rem_after >= 2:
                                ff_p = mx.stack(
                                    [mx.concatenate([p[k] for p in pending_tps], axis=0) for k in tap_keys],
                                    axis=1,
                                )[None]
                                pending_tps.clear()
                                dr.append_ctx(ff_p)
                                last_feat = ff_p[:, -1:]
                                packed_cur = dr.draft_and_pack_tree_dink(
                                    y_cur, last_feat, cost_arr, dink_cfg,
                                    max_depth=D_DYN, W_max=W_MAX,
                                )
                                mx.async_eval(*packed_cur)
                            else:
                                pending_tps.clear()
                        tok = int(y_cur.item())
                        stats["cycles"] += 1
                        stats["tokens"] += 1
                        yield tok, None, False
                        n += 1
                        if i + 1 < n_burst:
                            y_cur = y_nxt
                        else:
                            anchor = y_cur
                    if want_burst and packed_cur is not None:
                        want_burst = False
                        reprobe_d = True
                        cyc_since_burst = 0
                        cur_burst_d = min(dyn_burst_max, cur_burst_d * 2)
                        off_cpu = int(dec.caches[0][0][0].offset)
                        dec.ensure_gpu_cap(max_tokens - n + 32)
                        dr._ensure_cap(dr.n + max_tokens - n + 32)
                    continue

                while len(inflight_q) < PA_DEPTH and not want_burst:
                    if not inflight_q:
                        src_pkt = packed_cur
                        if src_pkt is None:
                            break
                        mr, mf = dec.get_gpu_meta(staged=False)
                        dp = _pos_arr(dr.n)
                    else:
                        last_cyc = inflight_q[-1]
                        src_pkt = last_cyc[8]
                        if src_pkt is None:
                            break
                        mr, mf, dp = last_cyc[5], last_cyc[6], last_cyc[7]
                    q_len = len(inflight_q)
                    ok_g, binfo_g, ring_wrap = _can_launch_gpu(q_len, off_cpu, dr.n)
                    if not ok_g:
                        break
                    launch_draft = (not want_burst) and (n + (q_len + 1) * W_MAX + 2 <= max_tokens)
                    inflight_q.append(
                        _launch_gpu_cycle_dyn(
                            src_pkt, mr, mf, dp, binfo_g,
                            ring_may_wrap=ring_wrap,
                            launch_draft=launch_draft,
                        )
                    )
                    if q_len == 0:
                        packed_cur = None

                if inflight_q:
                    (
                        emitted_c, keep_arr_c, w_act_c, anc_nxt, last_feat_nxt,
                        _, _, _, p_nxt,
                    ) = inflight_q.popleft()
                    mx.eval(emitted_c, keep_arr_c, w_act_c)
                    keep = int(keep_arr_c.item())
                    w_used = min(W_MAX, max(2, int(w_act_c.item())))
                    em = emitted_c.tolist()[:keep]
                    dec.sync_kv_offsets(keep, staged=False)
                    dr.n += keep
                    off_cpu += keep
                    stats["cycles"] += 1
                    stats["accepted"] += keep - 1

                    c_used = costs[w_used]
                    if reprobe_d:
                        etok_cpu = 0.5 * etok_cpu + 0.5 * keep
                        ems_cpu = 0.5 * ems_cpu + 0.5 * c_used
                    else:
                        etok_cpu += alpha_dyn * (keep - etok_cpu)
                        ems_cpu += alpha_dyn * (c_used - ems_cpu)
                    cyc_since_burst += 1
                    if not want_burst and force_dyn_w == 0 and drop_ratio_dyn > 0.0 and n >= 16:
                        if (reprobe_d or cyc_since_burst >= min_tree_dyn) and (etok_cpu / max(ems_cpu, 1.0)) < rate_w1_dyn * drop_ratio_dyn:
                            want_burst = True
                            reprobe_d = False
                        elif (etok_cpu / max(ems_cpu, 1.0)) >= rate_w1_dyn * drop_ratio_dyn:
                            cur_burst_d = dyn_burst_init
                            reprobe_d = False

                    for j in range(keep):
                        stats["tokens"] += 1
                        yield em[j], None, (j < keep - 1)
                        n += 1
                        if n >= max_tokens:
                            break

                    if not inflight_q:
                        anchor = anc_nxt
                        last_feat = last_feat_nxt
                        if want_burst:
                            packed_cur = None
                        elif p_nxt is not None:
                            packed_cur = p_nxt
                        elif n + 2 <= max_tokens:
                            packed_cur = dr.draft_and_pack_tree_dink(
                                anchor, last_feat, cost_arr, dink_cfg,
                                max_depth=D_DYN, W_max=W_MAX,
                            )
                            mx.async_eval(*packed_cur)
                        else:
                            packed_cur = None
                    continue

                # Synchronous fallback across regime boundaries (e.g. s=1024) or tail (rem < W_MAX)
                blk_c, par_c, dep_c, _, _, w_act_c = packed_cur
                packed_cur = None
                mx.eval(blk_c, par_c, dep_c, w_act_c)
                w_c = min(W_MAX, max(1, int(w_act_c.item())), max_tokens - n)
                if w_c <= 1:
                    y_cur, tp = dec.step_with_taps(anchor, tap_visits=tap_set)
                    f_1 = _feats_from_dict(tp, tap_keys)
                    dr.append_ctx(f_1)
                    mx.eval(y_cur, dr.ck, dr.cv, *dec.kv_eval_targets(), dec.conv)
                    tok = int(y_cur.item())
                    stats["cycles"] += 1
                    stats["tokens"] += 1
                    yield tok, None, False
                    n += 1
                    off_cpu = int(dec.caches[0][0][0].offset)
                    anchor = y_cur
                    last_feat = f_1[:, -1:]
                    if n + 2 <= max_tokens:
                        packed_cur = dr.draft_and_pack_tree_dink(
                            anchor, last_feat, cost_arr, dink_cfg,
                            max_depth=D_DYN, W_max=W_MAX,
                        )
                        mx.async_eval(*packed_cur)
                    continue

                par_l = par_c[:w_c].tolist()
                dep_l = dep_c[:w_c].tolist()
                _, ca_f, tm_f = build_tree_tables(par_l, dep_l)
                t, kv_news, conv_ws, tap_buf = dec.verify_w(
                    blk_c[:w_c],
                    tap_visits=tap_check,
                    depths_arr=dep_c[:w_c],
                    max_depth=max(dep_l),
                    conv_anc=ca_f,
                    tree_meta=tm_f,
                )
                mx.eval(t, blk_c)
                tl = t.tolist()
                bl = blk_c[:w_c].tolist()
                children = [[] for _ in range(w_c)]
                for s in range(1, w_c):
                    children[par_l[s]].append((s, bl[s]))
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
                a = keep - 1
                stats["cycles"] += 1
                stats["accepted"] += a

                n_after = n + keep
                if n_after < max_tokens:
                    anchor_next = t[cur_slot : cur_slot + 1]
                    dec.commit_w(kv_news, conv_ws, keep, path=path)
                    off_cpu = int(dec.caches[0][0][0].offset)
                    f_w = _feats_from_dict(tap_buf, tap_keys)
                    f_keep = f_w[:, :keep] if path == list(range(keep)) else f_w[:, mx.array(path, mx.int32)]
                    dr.append_ctx(f_keep)
                    last_feat = f_keep[:, -1:]
                    if n_after + 2 <= max_tokens:
                        packed_cur = dr.draft_and_pack_tree_dink(
                            anchor_next, last_feat, cost_arr, dink_cfg,
                            max_depth=D_DYN, W_max=W_MAX,
                        )
                        mx.async_eval(*dec.kv_eval_targets(), dec.conv, dr.ck, dr.cv, *packed_cur)
                    else:
                        mx.async_eval(*dec.kv_eval_targets(), dec.conv, dr.ck, dr.cv)
                    anchor = anchor_next

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
            return

        def choose_next_w(rem):
            if rem <= 1:
                return 1
            if force_w is not None:
                return max(1, min(force_w, B, rem))
            if (is_reprobe or tree_cycles >= min_tree_cycles) and (ema_tpc / cost_tree(w_cur) < rate_w1 * drop_ratio):
                return 1
            if w_cur >= hi_w and ema_tpc >= dn_thresh:
                w = hi_w
            elif ema_tpc >= up_thresh:
                w = hi_w
            else:
                w = w_init
            return max(2, min(w, B, rem))

        def _launch_gpu_cycle(packed, meta_r, meta_f, dr_pos, next_w, next_d, binfo, ring_may_wrap):
            blk_c, par_c, dep_c, ca_c, tm_c, w_c, d_c = packed
            walk_res, kv_news, conv_ws, tap_buf = dec.verify_w_gpu(
                blk_c,
                meta_r=meta_r,
                meta_f=meta_f,
                depths_arr=dep_c,
                conv_anc=ca_c,
                tree_meta=tm_c,
                max_depth=d_c,
                binfo=binfo,
                tau_cap=tau_cap,
                tap_visits=tap_check,
                par_arr=par_c,
                dr_pos=dr_pos,
                use_stage=use_stage,
            )
            if use_stage:
                (
                    path_pad, emitted, keep_arr, last_slot, anc_nxt,
                    slot_r_old, end_f_old, dr_pos_old, meta_r_nxt, meta_f_nxt, dr_pos_nxt, st_pos_old,
                ) = walk_res
                dec.commit_w_stage_gpu(kv_news, conv_ws, path_pad, last_slot, st_pos_old, w_c)
            else:
                (
                    path_pad, emitted, keep_arr, last_slot, anc_nxt,
                    slot_r_old, end_f_old, dr_pos_old, meta_r_nxt, meta_f_nxt, dr_pos_nxt,
                ) = walk_res
                eval_a = [
                    emitted, keep_arr, path_pad, last_slot, anc_nxt,
                    slot_r_old, end_f_old, dr_pos_old, meta_r_nxt, meta_f_nxt, dr_pos_nxt,
                    *kv_news,
                ]
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
            last_feat_nxt = f_w[:, last_slot]
            dr.append_ctx_gpu(f_pad, dr_pos_old)

            eval_b = [*dec.kv_eval_targets(staged=use_stage)]
            if use_stage:
                eval_b.extend((emitted, keep_arr, meta_r_nxt, meta_f_nxt, dr_pos_nxt))
            for cs in dec.conv:
                eval_b.extend(cs)
            eval_b.extend(dr.ck)
            eval_b.extend(dr.cv)
            if next_w >= 2:
                blk_n, par_n, dep_n, ca_n, tm_n = dr.draft_and_pack_tree(
                    anc_nxt, last_feat_nxt, max_depth=next_d, W=next_w, pos_arr=dr_pos_nxt,
                )
                packed_nxt = (blk_n, par_n, dep_n, ca_n, tm_n, next_w, next_d)
                eval_b.extend((blk_n, par_n, dep_n, ca_n, tm_n))
            else:
                packed_nxt = None
            mx.async_eval(*eval_b)
            return (
                emitted, keep_arr, w_c, d_c, anc_nxt, last_feat_nxt,
                meta_r_nxt, meta_f_nxt, dr_pos_nxt, packed_nxt,
            )

        packed_cur = None
        inflight = None
        w0 = choose_next_w(max_tokens - 1) if max_tokens > 1 else 1
        if w0 >= 2:
            d0 = d_for_w(w0)
            blk_n, par_n, dep_n, ca_n, tm_n = dr.draft_and_pack_tree(anchor, last_feat, max_depth=d0, W=w0)
            mx.async_eval(blk_n, par_n, dep_n, ca_n, tm_n)
            packed_cur = (blk_n, par_n, dep_n, ca_n, tm_n, w0, d0)

        yield y_tok, None, False
        n = 1
        pending_tps = []

        try:
            while n < max_tokens:
                if inflight is None and packed_cur is None:
                    rem = max_tokens - n
                    n_burst = rem if force_w == 1 else min(rem, cur_burst)
                    y_cur, tp = dec.step_with_taps(anchor, tap_visits=tap_set)
                    mx.async_eval(y_cur, *tp.values())
                    pending_tps.append(tp)
                    for i in range(n_burst):
                        if i + 1 < n_burst:
                            y_nxt, tp_nxt = dec.step_with_taps(y_cur, tap_visits=tap_set)
                            mx.async_eval(y_nxt, *tp_nxt.values())
                            pending_tps.append(tp_nxt)
                        else:
                            rem_after = max_tokens - (n + 1)
                            if rem_after > 1 and force_w != 1:
                                ff_p = mx.stack(
                                    [mx.concatenate([p[k] for p in pending_tps], axis=0) for k in tap_keys],
                                    axis=1,
                                )[None]
                                pending_tps.clear()
                                dr.append_ctx(ff_p)
                                last_feat = ff_p[:, -1:]
                                w_probe = min(w_init, B, rem_after)
                                d_probe = d_for_w(w_probe)
                                blk_n, par_n, dep_n, ca_n, tm_n = dr.draft_and_pack_tree(y_cur, last_feat, max_depth=d_probe, W=w_probe)
                                mx.async_eval(blk_n, par_n, dep_n, ca_n, tm_n)
                                packed_cur = (blk_n, par_n, dep_n, ca_n, tm_n, w_probe, d_probe)
                            else:
                                pending_tps.clear()
                        tok = int(y_cur.item())
                        stats["cycles"] += 1
                        stats["tokens"] += 1
                        yield tok, None, False
                        n += 1
                        if i + 1 < n_burst:
                            y_cur = y_nxt
                        else:
                            anchor = y_cur
                    if force_w != 1 and packed_cur is not None:
                        is_reprobe = True
                        ema_tpc = max(ema_tpc, 1.82)
                        tree_cycles = 0
                        cur_burst = min(explore_every, cur_burst * 2)
                        off_cpu = int(dec.caches[0][0][0].offset)
                        if use_pipe_ahead:
                            _ensure_lazy_gpu_cap(max_tokens - n, off_cpu)
                            dr._ensure_cap(dr.n + max_tokens - n + 16)
                    continue

                if inflight is None and packed_cur is not None and use_pipe_ahead:
                    blk_c, par_c, dep_c, ca_c, tm_c, w_c, d_c = packed_cur
                    ok_gpu = False
                    binfo = None
                    if n + w_c <= max_tokens:
                        if not dec._kv_stacked:
                            _ensure_lazy_gpu_cap(max_tokens - n, off_cpu)
                            off_cpu = int(dec.caches[0][0][0].offset)
                        elif use_stage and dec._kv_ring_st is None:
                            dec.init_kv_stage()
                        if dec._kv_stacked and dr.ck[0] is not None:
                            if use_stage and dec._kv_st_len + w_c > KV_STAGE_CAP:
                                dec.flush_kv_stage()
                            ok_b, binfo = dec._cw_safe_bounds_range(off_cpu, off_cpu, d_c)
                            if ok_b:
                                cap_r, cap_f, max_size_r = binfo[4], binfo[5], binfo[6]
                                if off_cpu + w_c > cap_f or (cap_r < max_size_r and off_cpu + w_c > cap_r):
                                    dec.ensure_gpu_cap(max_tokens - n + 16)
                                    ok_b, binfo = dec._cw_safe_bounds_range(off_cpu, off_cpu, d_c)
                                if ok_b:
                                    cap_r, cap_f, max_size_r = binfo[4], binfo[5], binfo[6]
                                    if (
                                        off_cpu + w_c <= cap_f
                                        and (cap_r == max_size_r or off_cpu + w_c <= cap_r)
                                        and dr.n + w_c + B <= int(dr.ck[0].shape[2])
                                    ):
                                        ok_gpu = True
                    if ok_gpu:
                        rem_min = max_tokens - (n + w_c)
                        w_nxt = 0 if (is_reprobe or rem_min < 2) else choose_next_w(rem_min)
                        d_nxt = d_for_w(w_nxt) if w_nxt >= 2 else 1
                        meta_r0, meta_f0 = dec.get_gpu_meta(staged=use_stage)
                        dr_pos0 = _pos_arr(dr.n)
                        inflight = _launch_gpu_cycle(
                            packed_cur, meta_r0, meta_f0, dr_pos0, w_nxt, d_nxt, binfo,
                            ring_may_wrap=(off_cpu + w_c > binfo[6]),
                        )
                        packed_cur = None

                if inflight is not None:
                    (
                        emitted_c, keep_arr_c, w_c, d_c, anc_nxt, last_feat_nxt,
                        meta_r_nxt, meta_f_nxt, dr_pos_nxt, p_nxt,
                    ) = inflight
                    next_inflight = None
                    if not is_reprobe and p_nxt is not None:
                        w_next = p_nxt[5]
                        d_next = p_nxt[6]
                        if (
                            n + w_c + w_next <= max_tokens
                            and dec._kv_stacked
                            and dr.ck[0] is not None
                            and (not use_stage or dec._kv_st_len + w_c + w_next <= KV_STAGE_CAP)
                        ):
                            end_lo = off_cpu + 1
                            end_hi = off_cpu + w_c
                            ok_b2, binfo2 = dec._cw_safe_bounds_range(end_lo, end_hi, d_next)
                            if ok_b2:
                                cap_r2, cap_f2, max_size_r2 = binfo2[4], binfo2[5], binfo2[6]
                                if (
                                    end_hi + w_next <= cap_f2
                                    and (cap_r2 == max_size_r2 or end_hi + w_next <= cap_r2)
                                    and dr.n + w_c + w_next + B <= int(dr.ck[0].shape[2])
                                ):
                                    rem_after_2 = max_tokens - (n + w_c + w_next)
                                    w_2 = choose_next_w(rem_after_2) if rem_after_2 >= 2 else 0
                                    if w_2 < 2:
                                        w_2 = 0
                                    d_2 = d_for_w(w_2) if w_2 >= 2 else 1
                                    next_inflight = _launch_gpu_cycle(
                                        p_nxt, meta_r_nxt, meta_f_nxt, dr_pos_nxt, w_2, d_2, binfo2,
                                        ring_may_wrap=(end_hi + w_next > max_size_r2),
                                    )

                    mx.eval(emitted_c, keep_arr_c)
                    keep = int(keep_arr_c.item())
                    em = emitted_c.tolist()[:keep]
                    dec.sync_kv_offsets(keep, staged=use_stage)
                    dr.n += keep
                    off_cpu += keep
                    stats["cycles"] += 1
                    stats["accepted"] += keep - 1
                    w_cur = w_c
                    ema_tpc = (1.0 - alpha_tpc) * ema_tpc + alpha_tpc * keep
                    tree_cycles += 1
                    if ema_tpc / cost_tree(w_cur) >= rate_w1 * drop_ratio:
                        cur_burst = burst_init
                    w_want = choose_next_w(max_tokens - (n + keep))
                    is_reprobe = False

                    for j in range(keep):
                        stats["tokens"] += 1
                        yield em[j], None, (j < keep - 1)
                        n += 1
                        if n >= max_tokens:
                            break

                    if next_inflight is not None:
                        inflight = next_inflight
                    else:
                        inflight = None
                        anchor = anc_nxt
                        last_feat = last_feat_nxt
                        if n < max_tokens:
                            if w_want == 1:
                                packed_cur = None
                                tree_cycles = 0
                            elif p_nxt is not None and p_nxt[5] == w_want and p_nxt[5] <= (max_tokens - n):
                                packed_cur = p_nxt
                            else:
                                d_want = d_for_w(w_want)
                                blk_n, par_n, dep_n, ca_n, tm_n = dr.draft_and_pack_tree(
                                    anchor, last_feat, max_depth=d_want, W=w_want,
                                )
                                mx.async_eval(blk_n, par_n, dep_n, ca_n, tm_n)
                                packed_cur = (blk_n, par_n, dep_n, ca_n, tm_n, w_want, d_want)
                    continue

                blk_c, par_c, dep_c, ca_c, tm_c, w_c, d_c = packed_cur
                packed_cur = None
                t, kv_news, conv_ws, tap_buf = dec.verify_w(
                    blk_c,
                    tap_visits=tap_check,
                    depths_arr=dep_c,
                    max_depth=d_c,
                    conv_anc=ca_c,
                    tree_meta=tm_c,
                )
                mx.eval(t, blk_c, par_c)

                tl = t.tolist()
                blk_l = blk_c.tolist()
                par_l = par_c.tolist()
                children = [[] for _ in range(w_c)]
                for s in range(1, w_c):
                    children[par_l[s]].append((s, blk_l[s]))

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
                a = keep - 1
                stats["cycles"] += 1
                stats["accepted"] += a

                w_cur = w_c
                ema_tpc = (1.0 - alpha_tpc) * ema_tpc + alpha_tpc * keep
                tree_cycles += 1
                if ema_tpc / cost_tree(w_cur) >= rate_w1 * drop_ratio:
                    cur_burst = burst_init

                n_after = n + keep
                w_next = choose_next_w(max_tokens - n_after) if n_after < max_tokens else 1
                is_reprobe = False
                if n_after < max_tokens:
                    anchor_next = t[cur_slot : cur_slot + 1]
                    dec.commit_w(kv_news, conv_ws, keep, path=path)
                    off_cpu = int(dec.caches[0][0][0].offset)
                    f_w = _feats_from_dict(tap_buf, tap_keys)
                    f_keep = f_w[:, :keep] if path == list(range(keep)) else f_w[:, mx.array(path, mx.int32)]
                    dr.append_ctx(f_keep)
                    last_feat = f_keep[:, -1:]
                    if w_next >= 2:
                        d_next = d_for_w(w_next)
                        blk_n, par_n, dep_n, ca_n, tm_n = dr.draft_and_pack_tree(
                            anchor_next, last_feat, max_depth=d_next, W=w_next,
                        )
                        mx.async_eval(*dec.kv_eval_targets(), dec.conv, blk_n, par_n, dep_n, ca_n, tm_n)
                        packed_cur = (blk_n, par_n, dep_n, ca_n, tm_n, w_next, d_next)
                    else:
                        tree_cycles = 0
                        mx.async_eval(*dec.kv_eval_targets(), dec.conv, dr.ck, dr.cv)
                    anchor = anchor_next

                for j in range(1, keep):
                    stats["tokens"] += 1
                    yield blk_l[path[j]], None, True
                    n += 1
                    if n >= max_tokens:
                        break
                if n < max_tokens:
                    stats["tokens"] += 1
                    yield bonus_tok, None, False
                    n += 1
        finally:
            if use_pipe_ahead and use_stage:
                dec.flush_kv_stage()
        return

    yield y_tok, None, False
    n = 1
    if n >= max_tokens:
        return

    q = [1.0, 0.70, 0.70, 0.70, 0.58, 0.58, 0.55, 0.55]
    # Calibrated 512S2 p1 causal verify + FastDrafter cycle latency (ms) for W in [1,2,3,4,5,6,8]
    cost = lambda L: fscale * float(np.interp(L, [1, 2, 3, 4, 5, 6, 8], [4.95, 7.80, 8.65, 9.60, 10.35, 11.20, 13.70]))
    probe_w = int(os.environ.get("LOOPLING_MTP_PROBE_W", "5"))
    explore_every = int(os.environ.get("LOOPLING_MTP_EXPLORE", "32"))
    alpha_ema = 0.15

    def pick():
        bw, best = 1, 1.0 / (4.95 * fscale)
        p, acc = 1.0, 1.0
        for W in range(2, B + 1):
            p *= q[W - 1]
            acc += p
            g = acc / cost(W)
            if g > best:
                best, bw = g, W
        return bw

    since = 1 << 30

    def choose_w(rem):
        nonlocal since
        if wmode.isdigit():
            w = int(wmode)
        elif since >= explore_every:
            since = 0
            w = probe_w
        else:
            w = pick()
        if w >= probe_w:
            since = 0
        if rem < w:
            w = max(1, rem)
        return w

    tap_check = tap_set | {"F"}
    pending_fs = []
    d_ready = None
    W_ready = None

    while n < max_tokens:
        if d_ready is not None:
            d, W = d_ready, W_ready
            d_ready = W_ready = None
        else:
            W = choose_w(max_tokens - n)
            if W == 1:
                N = min(8, max_tokens - n)
                y_cur, tp = dec.step_with_taps(anchor, tap_visits=tap_set)
                mx.async_eval(y_cur, *tp.values())
                pending_fs.append(_feats_from_dict(tp, tap_keys))
                for i in range(N):
                    if i < N - 1:
                        y_nxt, tp_nxt = dec.step_with_taps(y_cur, tap_visits=tap_set)
                        mx.async_eval(y_nxt, *tp_nxt.values())
                        pending_fs.append(_feats_from_dict(tp_nxt, tap_keys))
                    tok = int(y_cur.item())
                    stats["cycles"] += 1
                    stats["tokens"] += 1
                    yield tok, None, False
                    n += 1
                    if i < N - 1:
                        y_cur = y_nxt
                since += N
                anchor = y_cur
                continue

            if pending_fs:
                ff_p = mx.concatenate(pending_fs, axis=1)
                dr.append_ctx(ff_p)
                last_feat = ff_p[:, -1:]
                pending_fs.clear()
            d = dr.draft(anchor, last_feat, n_draft=W - 1)
            mx.async_eval(d)

        block = mx.concatenate([anchor, d])
        t, kv_news, conv_ws, tap_buf = dec.verify_w(block, tap_visits=tap_check)
        eval_list = [d, t] + kv_news
        for cw in conv_ws:
            eval_list.extend(cw)
        eval_list.extend(tap_buf.values())
        mx.eval(*eval_list)

        dl, tl = d.tolist(), t.tolist()
        a = 0
        while a < W - 1 and dl[a] == tl[a]:
            a += 1
        for j in range(1, min(a + 1, W - 1) + 1):
            q[j] += alpha_ema * ((1.0 if j <= a else 0.0) - q[j])
        keep = a + 1
        since += keep
        stats["cycles"] += 1
        stats["accepted"] += a

        n_after = n + keep
        if n_after < max_tokens:
            anchor_next = t[a : a + 1]
            f_w = _feats_from_dict(tap_buf, tap_keys)
            f_keep = f_w[:, :keep]
            W_next = choose_w(max_tokens - n_after)
            if W_next >= 2:
                dr.append_ctx(f_keep)
                last_feat = f_keep[:, -1:]
                d_ready = dr.draft(anchor_next, last_feat, n_draft=W_next - 1)
                mx.async_eval(d_ready)
                W_ready = W_next
                dec.commit_w(kv_news, conv_ws, keep)
            else:
                dec.commit_w(kv_news, conv_ws, keep)
                pending_fs.append(f_keep)
                last_feat = f_keep[:, -1:]
            anchor = anchor_next

        for j in range(keep):
            tok = dl[j] if j < a else tl[a]
            stats["tokens"] += 1
            yield tok, None, j < a
            n += 1
            if n >= max_tokens:
                break


def _legacy_mtp_generate_step(prompt, model, *, max_tokens=256, sampler=None, prefill_step_size=2048, prompt_cache=None, **kw):
    """Greedy self-speculative decoding with the official drafter. Yields (token, logprobs, from_draft)."""
    from mlx_lm.models.cache import make_prompt_cache

    tm = model.model.language_model
    passes = tm.args.loop_config["num_passes"]
    taps = parse_taps(os.environ.get("LOOPLING_MTP_TAPS", default_taps(passes)), passes)
    tm.tap_visits = {int(t[1:]) for t in taps if t[0] == "i"}
    tm.tap_buf = {}
    dr = Drafter(model)
    cache = prompt_cache or make_prompt_cache(model)
    prepare_record(cache)
    B = dr.block
    stats = getattr(model, "mtp_stats", None)
    if stats is None:
        stats = model.mtp_stats = {"cycles": 0, "accepted": 0, "tokens": 0}
    stats.update(cycles=0, accepted=0, tokens=0)

    def fwd(toks):
        lg = model(toks[None], cache=cache)
        return lg, feats_from_buf(tm, taps)

    # prefill all but the last prompt token in chunks, then the last token alone (as generate_step)
    n_p = prompt.size
    pos = 0
    while n_p - pos > 1:
        m = min(prefill_step_size, n_p - 1 - pos)
        _, f = fwd(prompt[pos : pos + m])
        dr.append_ctx(f)
        mx.eval([c.state for c in cache], dr.ck, dr.cv)
        pos += m
        mx.clear_cache()
    lg, f = fwd(prompt[pos:])
    dr.append_ctx(f)
    last_feat = f[:, -1:]
    y = mx.argmax(lg[:, -1, :], axis=-1)
    mx.eval(y)
    logp = lg[0, -1] - mx.logsumexp(lg[0, -1])
    yield int(y.item()), logp, False
    anchor = y
    n = 0
    wmode = os.environ.get("LOOPLING_MTP_W", "auto")
    q = [0.6] * B  # q[j]: P(draft j accepted | drafts < j accepted), EMA; verify width chosen to maximize tokens/ms
    fscale = (12 + 12 * passes) / 48.0  # target cost relative to 3 passes
    cost = lambda L: fscale * float(np.interp(L, [1, 2, 4, 6, 8], [27.9, 33.4, 38.3, 43.2, 47.1])) + 7.0  # measured 512S1, p3 (ms)

    def pick():
        bw, p, acc = 1, 1.0, 1.0
        best = 1.0 / (26.5 * fscale)  # plain pipelined decode: no drafter
        for W in range(2, B + 1):
            p *= q[W - 1]
            acc += p
            g = acc / cost(W)
            if g > best:
                best, bw = g, W
        return bw

    since = 1 << 30  # tokens since the last full-width exploration cycle (forces one at the start)
    while n < max_tokens:
        W = int(wmode) if wmode.isdigit() else (B if since >= 64 else pick())
        if W == 1:
            # plain pipelined decode burst (as generate_step); feeds the drafter's context afterwards
            N = 8
            lg, f = fwd(anchor)
            y = mx.argmax(lg[:, -1], axis=-1)
            mx.async_eval(y, f)
            fs = [f]
            for i in range(N):
                if i < N - 1:
                    lg, f = fwd(y)
                    y2 = mx.argmax(lg[:, -1], axis=-1)
                    mx.async_eval(y2, f)
                    fs.append(f)
                tok = int(y.item())
                stats["cycles"] += 1
                stats["tokens"] += 1
                yield tok, None, False
                n += 1
                if i < N - 1:
                    y = y2
            since += N
            anchor = y
            ff = mx.concatenate(fs, axis=1)
            dr.append_ctx(ff)
            last_feat = ff[:, -1:]
            mx.eval(dr.ck, dr.cv)
            continue
        if W == B:
            since = 0
        d = dr.draft(anchor, last_feat)  # (7,) lazy
        block = mx.concatenate([anchor, d[: W - 1]])
        lg, f = fwd(block)
        t = mx.argmax(lg[0], axis=-1)  # (W,) greedy target after each block position
        mx.eval(d, t, f)
        dl, tl = d.tolist(), t.tolist()
        a = 0
        while a < W - 1 and dl[a] == tl[a]:
            a += 1
        for j in range(1, min(a + 1, W - 1) + 1):
            q[j] += 0.1 * ((1.0 if j <= a else 0.0) - q[j])
        stats["cycles"] += 1
        stats["accepted"] += a
        keep = a + 1
        since += keep
        rollback(cache, W, keep)
        dr.append_ctx(f[:, :keep])
        last_feat = f[:, keep - 1 : keep]
        mx.eval(dr.ck, dr.cv)
        anchor = t[a : a + 1]
        for j in range(a + 1):
            tok = dl[j] if j < a else tl[a]
            stats["tokens"] += 1
            yield tok, None, j < a
            n += 1
        if n % 256 == 0:
            mx.clear_cache()


def mtp_generate_step(
    prompt,
    model,
    *,
    max_tokens=256,
    sampler=None,
    prefill_step_size=2048,
    prompt_cache=None,
    full_prompt=None,
    eos_token_ids=None,
    checkpoint_callback=None,
    segment_ends=None,
    cache_state=None,
    prompt_progress_callback=None,
    **kw,
):
    """Greedy self-speculative decoding with the official drafter (or Splash DFlash2 when LOOPLING_MTP=dflash2). Yields (token, logprobs, from_draft)."""
    if os.environ.get("LOOPLING_MTP") == "dflash2":
        from mlx_lm.models.inkling2.dflash2 import dflash2_exact_generate_step

        return dflash2_exact_generate_step(
            prompt,
            model,
            max_tokens=max_tokens,
            sampler=sampler,
            prefill_step_size=prefill_step_size,
            prompt_cache=prompt_cache,
            full_prompt=full_prompt,
            eos_token_ids=eos_token_ids,
            checkpoint_callback=checkpoint_callback,
            segment_ends=segment_ends,
            cache_state=cache_state,
            prompt_progress_callback=prompt_progress_callback,
            **kw,
        )
    from mlx_lm.models.inkling2.batchdecode import BatchDecoder, _FUSED, _PLAIN
    from mlx_lm.models.inkling2 import fastdecode as FD, layers as L
    from mlx_lm.models.inkling2.rows import visit as RV

    tm = model.model.language_model
    passes = tm.args.loop_config["num_passes"]
    use_exact = (
        os.environ.get("LOOPLING_MTP_EXACT", "1") not in ("0", "")
        and os.environ.get("LOOPLING_MTP_LEGACY", "0") in ("0", "")
        and not (tm.loopcd.enabled_for(passes) and tm.loopcd.mode != "hidden")
        and L._use_kernels()
        and {"pre", "k1", "post"}.issubset(_FUSED)
        and not _PLAIN
        and not RV.FREEZE
        and not RV.DUP
        and not RV.SYNC
        and RV.MOE
        and FD._FCONV
        and tm.args.hidden_size == 2304
        and all(FD._fused_b1_ok_layer(layer) for layer in tm.layers)
        and os.path.exists(os.path.join(MTP_DIR, "mtp.safetensors"))
    )
    if use_exact:
        return mtp_exact_generate_step(
            prompt, model, max_tokens=max_tokens, sampler=sampler,
            prefill_step_size=prefill_step_size, prompt_cache=prompt_cache, **kw,
        )
    return _legacy_mtp_generate_step(
        prompt, model, max_tokens=max_tokens, sampler=sampler,
        prefill_step_size=prefill_step_size, prompt_cache=prompt_cache, **kw,
    )
