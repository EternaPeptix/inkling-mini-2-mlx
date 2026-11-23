"""Lane e5 part 1: primitive inventory of one decode step (B=1, L=1) of Loopling p3.

Runs the fast-decode segmentation WITHOUT mx.compile (LOOPLING_COMPILE_DECODE=2) and with mx.compile globally disabled
(so the compiled _l2_clip / _softcapped_swiglu show up as their elementwise primitives), walks the lazy graph of selected
layer visits with mx.export_to_dot and prints the primitive sequence (topological order of the dot file) per case.

    LANE=e5 RHOST=512S2 scripts/r --gpu 'LOOPLING_COMPILE_DECODE=2 python bench/glue_inventory.py'

Output: ~/loopling-opt/runs/e5/inv/<case>.txt (primitive list) + a summary on stdout; also dtype facts.
"""
import collections
import io
import os
import re
import sys

os.environ["LOOPLING_COMPILE_DECODE"] = "2"
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.inkling2 import fastdecode as FD
from mlx_lm.models.inkling2 import layers as L

OUT = os.path.expanduser("~/loopling-opt/runs/e5/inv")
os.makedirs(OUT, exist_ok=True)
mx.disable_compile()

model, tok = load(os.path.expanduser("~/loopling/models/loopling-p3"))
TM = model.model.language_model
args = TM.args
core = args.loop_core
passes = args.loop_config["num_passes"]
cache = make_prompt_cache(model)
ids = tok.encode("Write a Python function that parses an ISO-8601 duration like P3DT4H12M into seconds.")
p = mx.array(ids)
model(p[None, :-1], cache=cache)
mx.eval([c.state for c in cache])
y = p[-1:]


def dot_prims(*outs):
    """Primitive labels of the lazy graph that produces `outs`, in dot order."""
    f = io.StringIO()
    mx.export_to_dot(f, *outs)
    labs = re.findall(r'label ="([^"]+)"', f.getvalue())
    return labs


def flat(x):
    if isinstance(x, (list, tuple)):
        for t in x:
            yield from flat(t)
    elif isinstance(x, dict):
        for t in x.values():
            yield from flat(t)
    elif isinstance(x, mx.array):
        yield x


def report(case, labs):
    c = collections.Counter(labs)
    with open(os.path.join(OUT, case + ".txt"), "w") as fh:
        fh.write("\n".join("%3d %s" % (i, l) for i, l in enumerate(labs)) + "\n")
    print("== %-34s %3d primitives: %s" % (case, len(labs), ", ".join("%s x%d" % kv for kv in c.most_common())), flush=True)


# ---------------------------------------------------------------- dtype facts
def dt(a):
    return "%s%s" % (str(a.dtype).split(".")[-1], list(a.shape))


L1, L5, L6, L7 = TM.layers[1], TM.layers[5], TM.layers[6], TM.layers[7]
print("embed_norm.weight", dt(TM.embed_norm.weight), "| norm.weight", dt(TM.norm.weight))
print("L1 input_layernorm.weight", dt(L1.input_layernorm.weight), "| L1 post_attention_layernorm", dt(L1.post_attention_layernorm.weight))
print("L5 q/k/v_input_layernorm", dt(L5.q_input_layernorm.weight), "| L5 post_attention_layernorm", dt(L5.post_attention_layernorm.weight))
print("q_l2_clip.scale", dt(L1.self_attn.q_l2_clip.scale), "rel_l2_clip.scale", dt(L1.self_attn.rel_l2_clip.scale))
print("rel_logits_proj.proj", dt(L1.self_attn.rel_logits_proj.proj), "| router w", dt(L1.mlp.gate.weight), "bias", dt(L1.mlp.gate.e_score_correction_bias))
print("msg gate down", dt(L7.q_message_gate.down_proj.weight), "forget", dt(L7.q_message_gate.forget_gate_proj.weight), "message", dt(L7.q_message_gate.message_gate_proj.weight))
print("conv1d.weight (sliding k)", dt(L1.self_attn.k_sconv.conv1d.weight), "| post_attn conv", dt(L1.post_attention_convolution.conv1d.weight))
print("dyn conv static_weight", dt(L5.self_attn.k_sconv.static_weight), "weight_proj", type(L5.self_attn.k_sconv.weight_proj).__name__)
print("queries: attention_input L1", dt(L1.attention_input.query), "L5", dt(L5.attention_input.query), "mlp_input", dt(L1.mlp_input.query),
      "final_input", dt(TM.final_input.query), "message_input", dt(TM.message_input.query))
print("heads: sliding", L1.self_attn.num_heads, L1.self_attn.num_key_value_heads, "full", L5.self_attn.num_heads, L5.self_attn.num_key_value_heads)
print("logit softcap", args.final_logit_softcapping, "unpadded", args.unpadded_vocab_size, "vocab", args.vocab_size)

# ---------------------------------------------------------------- the step, visit by visit
WANT = {  # visit index -> case name (pass 0 of the loop core = visits 6..17)
    0: "S_boundary_first_nomsg_nobs",      # layer 0: sliding, boundary, no attention_input, bs None
    1: "S_nonboundary_nomsg",              # layer 1
    5: "F_nonboundary_nomsg",              # layer 5: full attention
    6: "S_boundary_nomsg",                 # layer 6 pass 0 (has bs from prelude)
    7: "S_nonboundary_nomsg_loopcore",     # layer 7 pass 0
    11: "F_nonboundary_nomsg_loopcore",    # layer 11 pass 0
    18 + 0: "S_boundary_msg",              # layer 6 pass 1
    18 + 1: "S_nonboundary_msg",           # layer 7 pass 1
    18 + 5: "F_nonboundary_msg",           # layer 11 pass 1
}
# visit numbering: prelude 0..5, core pass0 6..17, pass1 18..29, pass2 30..41, tail 42..47
WANT = {0: WANT[0], 1: WANT[1], 5: WANT[5], 6: WANT[6], 7: WANT[7], 11: WANT[11], 12 + 6: WANT[18], 12 + 7: WANT[19], 12 + 11: WANT[23]}
full_ids = [i for i in range(args.num_hidden_layers) if i not in args.local_layer_ids]
print("full attention layers:", full_ids, "boundary layers:", [i for i in range(24) if i % args.n_layers_per_attn_res_block == 0])

caches = iter(cache)
visit = 0
x = TM.embed(y[None])
mx.eval(x)
print("embed out", dt(x))
step_ops = {}
step_ops["embed (embed_tokens + embed_norm)"] = dot_prims(TM.embed(y[None]))


def run(layers, h, bs, bank=None):
    global visit
    for i, layer in enumerate(layers):
        msgs = None if bank is None else TM.message_input(bank, i)
        if msgs is not None:
            if visit in WANT or True:
                step_ops.setdefault("message_input (per visit, pass>0)", dot_prims(msgs))
            mx.eval(msgs)
        c = next(caches)
        if visit in WANT:
            # prime the per-step caches (mask prep / tau) so the visit graph shows only the steady-state ops
            sliding = isinstance(layer, L.SlidingDecoderLayer)
            kv = c[0]
            h_o, bs_o = layer(h, bs, cache=c, conv_mask=None, messages=msgs)
            outs = list(flat([h_o, bs_o, list(c[1].cache), kv.keys, kv.values]))
            labs = dot_prims(*outs)
            report(WANT[visit], labs)
            mx.eval(outs)
        else:
            h_o, bs_o = layer(h, bs, cache=c, conv_mask=None, messages=msgs)
            mx.eval(h_o, *(flat(bs_o) if bs_o is not None else []))
        h, bs = h_o, bs_o
        mx.eval(h, *(flat(bs) if bs is not None else []))
        visit += 1
    return h, bs


h, bs = x, None
prelude = run(TM.layers[: core.start], h, None)
h, bs = prelude
for lp in range(passes):
    bank = None
    if lp > 0:
        bank = (bs[prelude[1].shape[0] + 1 :], L.unweighted_rms_norm(h.astype(mx.float32)))
        if lp == 1:
            step_ops["bank (pass>0): slice + unweighted rms"] = dot_prims(*bank)
        h, bs = prelude
    h, bs = run(TM.layers[core.start : core.stop], h, bs, bank)
h, bs = run(TM.layers[core.stop :], h, bs)
nr = L.unweighted_rms_norm(h.astype(mx.float32))
fi = TM.final_input(bs, nr)[..., 0, :]
step_ops["final_input (nr rms + depth_attention + slice)"] = dot_prims(TM.final_input(bs, L.unweighted_rms_norm(h.astype(mx.float32)))[..., 0, :])
hn = TM.norm(fi).astype(h.dtype)
step_ops["norm (RMSNorm w) + astype"] = dot_prims(TM.norm(fi).astype(h.dtype))
mx.eval(hn)
logits = model.lm_head(hn)
step_ops["lm_head (quantized)"] = dot_prims(model.lm_head(hn))
lf = logits.astype(mx.float32)
sc = mx.tanh(lf / 30.0) * 30.0
sl = sc[..., : args.unpadded_vocab_size]
step_ops["softcap + slice"] = dot_prims(sl)
mx.eval(sl)
lg = sl[:, -1, :]
lp = lg - mx.logsumexp(lg, keepdims=True)
step_ops["logprobs = logits - logsumexp (keepdims, all axes)"] = dot_prims(lp)
mx.eval(lp)
step_ops["greedy argmax(logprobs, -1)"] = dot_prims(mx.argmax(lp, axis=-1))
print("hidden after last layer", dt(h), "| bs", dt(bs), "| final_input out", dt(fi), "| hn", dt(hn), "| logits", dt(logits), "| sl", dt(sl), "| lg", dt(lg), "| lp", dt(lp))
for k, v in step_ops.items():
    report("step_" + re.sub(r"[^A-Za-z0-9]+", "_", k)[:40], v)

# per-step mask prep / tau (cached in FD._ctx / FD._tau, built lazily once per step per attention type)
print("FD._ctx keys (step-level mask prep, built once per step per (S,end,idx,R,window)):", list(FD._ctx.keys())[:4])
a = TM.layers[1].self_attn
kv = cache[1][0]
S, end = kv.keys.shape[2], FD._end(kv)
R = 1024
qp = L._positions(end - 1, 1)
kp = L._key_positions(kv, S, 1, end)
distance = qp.reshape(1) - kp.reshape(S)
clip = mx.clip(distance, 0, R - 1)
visible = (distance >= 0) & (kp.reshape(S) >= 0)
visible = visible & (distance < a.sliding_window)
use_bias = visible & (distance < R)
fill = mx.where(visible, 0.0, -mx.inf)
report("step_mask_prep_sliding", dot_prims(clip, use_bias, fill))
t = FD._full_tau(TM.layers[5], 20)
report("step_tau_full_layers", dot_prims(1.0 + 0.1 * mx.log(mx.maximum((mx.arange(1) + 20).astype(mx.float32) + 1, 1.0))))
