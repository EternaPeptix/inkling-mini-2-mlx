# LOOPLING_MTP: the official "DSpark/dflare" multi-token-prediction drafter of Loopling, ported to MLX.
# See docs/MTP.md for the math and the file:line citations into the vLLM reference.
import json
import os
import time

import mlx.core as mx
import numpy as np
import mlx.nn as nn

from mlx_lm.models.cache import KVCache, RotatingKVCache

MTP_DIR = os.environ.get("LOOPLING_MTP_DIR", os.path.expanduser("~/loopling/models/official-mtp"))
EPS = 1e-6


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
    return "i5,i23,i41,F"  # literal target_layer_ids; also best at 5 passes (docs/MTP.md)


def feats_from_buf(tm, taps):
    return mx.stack([tm.tap_buf[t if t in ("F", "f", "o") else int(t[1:])] for t in taps], axis=2)  # (1,T,ntap,H)


def mtp_generate_step(prompt, model, *, max_tokens=256, sampler=None, prefill_step_size=2048, prompt_cache=None, **kw):
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
