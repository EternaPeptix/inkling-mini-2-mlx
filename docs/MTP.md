# MTP: the official Loopling drafter ("DSpark" / "dflare") in MLX

Flag: `LOOPLING_MTP=1` (default OFF). Code: `vendor/mlx_lm/models/inkling2/mtp.py` (drafter, verify loop, cache rollback),
hooks in `vendor/mlx_lm/generate.py::stream_generate`, tap capture in `inkling2/model.py::TextModel._run_loop`,
conv-window recording in `inkling2/layers.py::DecoderLayer.__call__`. Other env: `LOOPLING_MTP_W` (`auto` default, or a fixed
verify width 2..8), `LOOPLING_MTP_TAPS` (tap list, default `i5,i23,i<6+12p-1>,F`), `LOOPLING_MTP_DIR`.

Reference (read only): `/Users/jeweled/.gemini/antigravity/brain/78201cfd-43f0-4f1b-ae82-c573f3cd7b11/scratch/patches-s6/`
`dspark_model.py` (M), `dspark_ops.py` (O), `speculator.py` (S), plus `attention_ops.py` (A) and
`../dec160/tree_spec/staged_s8v2/model_l2pf.py` (T, the target's tap convention). That reference is for Inkling-2-Small; the
Loopling checkpoint has the same drafter family with different sizes (hidden 2304, 3 layers, 24/8 heads, head_dim 128, MLP 9216,
block 8, rank 256, `target_layer_ids [5,23,41,48]`).

## What the drafter is

A 3-layer transformer that proposes a whole block of 7 tokens in ONE forward (not autoregressively), conditioned on the target's
own hidden states, followed by a small recurrent "Markov" head that makes the 7 greedy token choices sequentially-dependent.
The target then verifies `[anchor, d1..d7]` in one 8-token forward and keeps the longest matching prefix plus one bonus token.

## The math (our port; line numbers are in the reference files)

Notation: `P` = position of the anchor token (the last sampled token, not yet run through the target); `H=2304`, `T=4` taps.

1. **Target taps** (T:800-805, T:1040-1043; M:1001-1004 `fuse_weight` shape `(num_layers+1, len(target_layer_ids))`).
   The target hands the drafter 4 feature vectors per committed position: the hidden state *entering* visits 5, 23, 41 (the vLLM
   loop appends `partial` before running layer `i` when `i in aux_hidden_state_layers`) and, for id 48 (= number of layers, never
   hit by the loop), the final normed hidden `out` that feeds the LM head (T:1040-1043 returns `[*aux, out]`).
2. **Fuse + per-layer context** (M:1033-1035 `mix`, O:22-53 `_mix_rmsnorm_kernel`, M:1277-1279 softmax of `fuse_weight`):
   `ctx_l = RMSNorm_{dflare_norm_l}( sum_t softmax(fuse_weight[l])_t * feat_t )`, fp32 math, bf16 out. Row `l` for draft layer `l`,
   the extra last row for the Markov head. Learned fuse weights (softmax) e.g. layer 0 puts 0.68 on tap 3 (visit 23)...
3. **Context K/V** (M:599-643 `store_context_kv`): `k = k_proj_ctx(ctx_l)`, `v = v_proj_ctx(ctx_l)`; `k` gets per-head
   `RMSNorm(k_norm_ctx)` then neox RoPE at the committed position; `v` is raw. One K/V entry per committed target position.
4. **Block input** (M:1037-1043 `embed_block`): slot 0 = `embed_norm(embed(anchor))` (the *target's* normalized embedding),
   slots 1..7 = learned `mask_embed`. Block positions `P..P+7` (S:3507-3509: query pos = last_valid_pos+1+off).
5. **Layers** (M:716-762, M:498-596): standard pre-norm decoder, residual stream bf16:
   `h=RMSNorm(x)`; `q,k,v=q_proj,k_proj,v_proj(h)`; `q=RMSNorm(q_norm)`, `k=RMSNorm(k_norm)` per head; neox RoPE (theta 1e6, full
   head_dim) at block positions; attention of the 8 block queries over `[ctx K/V , block K/V]`, **bidirectional inside the block**,
   no relative bias / sink / gate (A:1315-1339 `block_decode_attention`, scale head_dim^-0.5); GQA 24/8; layers 0 and 2 are local
   (`local_layer_ids`): only the last 1024 context positions + the block (A: `window = sliding_window + block`), layer 1 global.
   `o_proj`; residual add; `RMSNorm(post_attention_layernorm)`; SwiGLU `down(silu(gate)*up)` (M:662-685); residual add.
   Final `RMSNorm(norm)`; slots 1..7 are the 7 hidden outputs (S:2944 `head_hidden[1:1+k]`; slot 0 is unused).
6. **Base logits** (M:1579): `base_i = target.lm_head(hid_i)` (shared LM head, raw, pre-softcap).
7. **Markov head** (M:765-880 `Inkling2MarkovHead`, O:595-610 `_markov_gate_kernel`, S:2881-2921 + S:2951-3000):
   * `f_rnn = RMSNorm_{markov.dflare_norm}(sum_t softmax(fuse_weight[-1])_t * feat_t(P-1))`: features of the last committed
     position (S:2905-2921).
   * `ctx_i = rnn_proj.W[:, 2r:] @ [hid_i ; f_rnn] + rnn_proj.b` (fp32), `r=256` (M:824-870 `context`).
   * Sequential over i=0..6 with `state_0 = 0`, `prev_0 = anchor`: `raw = ctx_i + rnn_proj.W[:, :2r] @ [state ; markov_w1[prev]]`
     (M:884-896 `step`); `gate, cand, out = raw[:r], raw[r:2r], raw[2r:]`;
     `state = sigmoid(gate)*state + (1-sigmoid(gate))*tanh(cand)`; `latent = tanh(out)` (O:600-610).
   * `bias_logits = lm_head(hidden_out(latent))`, folded once into a `(V, r)` matrix (M:798-815 `fold`). We fold with the quantized
     MLX lm_head applied to the `hidden_out` columns, bf16 (reference folds in fp64).
   * `draft_i = argmax( 30*tanh((base_i + bias_i)/30) )` over the unpadded vocab (O:573-582 `add_softcap`, M:1679-1690); `prev = draft_i`.
8. **Verify** (S:3507-3509, the bonus token logic): target forward on `[anchor, d1..d7]`; accept `d_{j+1}` while it equals the
   target's greedy token after position `j`; the target's token after the last accepted position is the bonus and becomes the next
   anchor. Context K/V and `f_rnn` for the accepted positions come from the same verify forward (taps are returned per position).

### Our deviations / uncertainty

* `layers.0.dflare_norm` is **absent from the checkpoint** (only layers 1, 2 and markov_head have one). The reference's `_RMSNorm`
  defaults to ones (M:297); we use ones. Ablation: using layer 1's norm instead drops teacher-forced accept length 1.69 -> 0.88 (ones is right or close).
* The reference deployment caps the global layer's window at 1024 (`INKLING_DRAFT_GLOBAL_WINDOW`, M:145-153); we use the true
  global attention (identical for contexts < 1024).
* The reference's bf16 rounding emulation inside the Markov gate (O:586-610) is only approximated (bf16 raw, bf16 state).
* The mapping of the reference's 48 target layers to Loopling's visits (below) is by analogy plus an acceptance test, not documented.

## Mapping `target_layer_ids [5,23,41,48]` onto Loopling's visits

Loopling at 3 passes has 6 prelude + 12x3 loop + 6 tail = **48 visits**, exactly the 48 layers the ids index (the official
checkpoint is a 3-pass model). The taps are the *inputs* of visit 5 (= end of the prelude), 23 (pass 1, core layer 11),
41 (pass 2, last core layer) and the final normed hidden. Evidence (teacher-forced drafter acceptance, 3 bench prompts, ground truth
= baseline greedy tokens, `bench/mtp_probe.py`; mean accepted drafts / block, 7 max):

| taps | code | prose | math | mean |
|---|---|---|---|---|
| **i5,i23,i41,F** (reference convention) | 1.11 | 0.40 | 3.55 | **1.69** |
| i5,i23,i41,o (raw output of last visit) | 0.89 | 0.36 | 3.31 | 1.52 |
| i5,i23,i41,f (final_input pre-norm) | 0.63 | 0.39 | 3.45 | 1.49 |
| i6,i24,i42,F (outputs instead of inputs) | 0.89 | 0.35 | 2.99 | 1.41 |
| i4,i22,i40,F | 0.84 | 0.43 | 3.08 | 1.45 |
| i5,i23,i41,i47 | 0.68 | 0.39 | 3.43 | 1.50 |
| no Markov bias (base logits only) | 0.58 | 0.23 | 1.16 | 0.65 |

The reference convention wins on every prompt where it matters, and removing the Markov head costs 60%, so both are in use.
